# po_app_react_agent_rag.py
from __future__ import annotations
from dotenv import load_dotenv
import os
import re
import urllib.parse
from typing import Annotated, Dict, Optional
from uuid import uuid4
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from helper.currency_helper import _extract_price_currency
from sqlalchemy import create_engine, text

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.tools import tool
from openai import AzureOpenAI
from helper.po_helpers import _insert_purchase_order, _generate_po_pdf  

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_ENDPOINT = "https://api.tavily.com/search"


# ---------- Prebuilt ReAct agent ----------
try:
    from langgraph.prebuilt import create_react_agent
except ImportError:
    create_react_agent = None  # fallback will be thrown below




load_dotenv()

# ---------- DB ----------
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DRIVER_PATH = "/opt/homebrew/opt/msodbcsql18/lib/libmsodbcsql.18.dylib"

ODBC_STR = (
    "Driver={{{driver_path}}};".format(driver_path=DRIVER_PATH) +
    f"Server=tcp:ramdemosql.database.windows.net,1433;"
    f"Database={DB_DATABASE};"
    f"Uid={DB_USER};"
    f"Pwd={DB_PASSWORD};"
    "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
)

_engine = None
def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    db_url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(ODBC_STR)
    _engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=1800)
    return _engine

# ---------- Utils ----------
def _normalize_product(name: str) -> str:
    """Normalize product name for lookup"""
    n = re.sub(r"\b(Product|laptop|pc|pcs|computer)s?\b$", "", name.strip(), flags=re.I)
    return n.strip().upper()

def interrupt_text(obj) -> str:
    """Return just the interrupt's value text."""
    if isinstance(obj, list):
        for x in obj:
            if hasattr(x, "value"):
                return str(x.value)
        return str(obj[0]) if obj else ""
    if hasattr(obj, "value"):
        return str(obj.value)
    if isinstance(obj, dict) and "value" in obj:
        return str(obj["value"])
    return str(obj)

# ---------- State ----------
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ---------- Azure RAG Tool ----------
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = "po-policy-index"
AOAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
EMBEDDING_MODEL = "text-embedding-ada-002"

if not AOAI_ENDPOINT:
    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set.")
aoai_client = AzureOpenAI(api_key=AOAI_KEY, azure_endpoint=AOAI_ENDPOINT, api_version="2024-05-01-preview")
if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY:
    raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY environment variables must be set.")


@tool
def search_products(query: str, filters: Optional[Dict] = None) -> Dict:
    
    """
    Search the web for purchasable products & pricing using Tavily.
    Returns up to 8 normalized candidates: {title, vendor, price, currency, url, source}.
    'filters' supports: currency (GBP/EUR/USD), max_results (<= 10), include_domains (list[str]).
    """
    print(f"[search_products] Searching for: {query}")
    if not TAVILY_API_KEY:
        return {"error": "TAVILY_API_KEY is not set", "candidates": []}

    filters = filters or {}
    currency = (filters.get("currency") or "£").upper()
    max_results = int(filters.get("max_results") or 8)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "max_results": max(1, min(max_results, 10)),
        "include_answer": False,
        "include_images": False,
        "topic": "general",
    }
    if currency in ("GBP", "EUR", "USD"):
        payload["currency"] = currency
    
    try:
        resp = requests.post(TAVILY_ENDPOINT, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            return {"error": f"Tavily error {resp.status_code}: {resp.text}", "candidates": []}
        data = resp.json()
    except Exception as e:
        return {"error": f"Tavily request failed: {e}", "candidates": []}

    results = data.get("results", []) or []
    candidates = []
    for r in results:
        url = r.get("url", "")
        title = r.get("title", "")
        content = r.get("content", "") or ""
        domain = urllib.parse.urlparse(url).netloc
        # Try to extract a price
        price, cur = _extract_price_currency(" ".join([title, content]), currency_hint=currency)
        candidates.append({
            "title": title[:200],
            "vendor": domain,
            "price": price,
            "currency": cur or currency,
            "url": url,
            "source": "tavily"
        })
    # Keep only entries with a price; sort by price asc
    priced = [c for c in candidates if isinstance(c.get("price"), (int, float))]
    priced.sort(key=lambda x: x["price"])
    # Fallback: if none had a price, return top unpriced pages
    final = priced if priced else candidates[:max_results]
    return {"query": query, "candidates": final[:max_results]}

@tool
def policy_lookup(query: str) -> str:
    """Retrieve top 3 relevant policy rules from Azure Search using REST API."""
    print(f"[policy_lookup] Searching for: {query}")

    # 1. Generate embedding
    emb = aoai_client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    query_vector = emb.data[0].embedding

    # 2. Build request body
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version=2024-11-01-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_KEY
    }
    payload = {
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": query_vector,
                "fields": "contentVector",
                "k": 5
            }
        ],
        "select": "title,content"
    }

    # 3. Make request
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"[policy_lookup] Error: {response.text}")
        return f"Policy lookup error: {response.text}"

    data = response.json()
    hits = []
    for doc in data.get("value", []):
        hits.append(f"- {doc.get('title', '')}: {doc.get('content', '')}")

    return "\n".join(hits) if hits else "No relevant policy found."


# ---------- Tools ----------
@tool
def check_availability(product: str) -> Dict:
    """
    Return availability and unit price for a product from procurement.Inventory.
    Case-insensitive, tolerant of 'Product' suffix.
    """
    print("check_availability called with product:", product)
    eng = _get_engine()
    norm = _normalize_product(product)
    with eng.connect() as conn:
        row = conn.execute(
            text("""
                SELECT Product, Available, UnitPrice
                FROM procurement.Inventory
                WHERE UPPER(Product) = :p
            """),
            {"p": norm},
        ).first()
        if not row:
            row = conn.execute(
                text("""
                    SELECT TOP 1 Product, Available, UnitPrice
                    FROM procurement.Inventory
                    WHERE UPPER(Product) LIKE :p
                    ORDER BY Product
                """),
                {"p": norm + "%"},
            ).first()

    if row:
        try:
            Product = row.Product
            available = int(row.Available)
            unit_price = float(row.UnitPrice)
        except Exception:
            Product, available, unit_price = row[0], int(row[1]), float(row[2])
        return {"product": Product, "available": available, "unit_price": unit_price}
    return {"product": norm, "available": 0, "unit_price": 0.0}

@tool
def purchaseorder_approval(product: str, quantity: int) -> Dict:
    """
    Decide on purchase order approval based on inventory, pricing, product and policy.
    """
    try:
        print(f"purchaseorder_approval called with product: {product}, quantity: {quantity}")

        # Always define these so they exist in all branches
        vendor_name: str = "N/A"
        vendor_url: str = "N/A"

        # Policy summary (RAG) — safe: returns text even on failure
        policy_summary = policy_lookup.invoke(
            f"purchase order approval rules for {product} quantity {quantity}"
        )

        eng = _get_engine()
        norm = _normalize_product(product)

        with eng.connect() as conn:
            row = conn.execute(
                text("SELECT Product, Available, UnitPrice FROM procurement.Inventory WHERE UPPER(Product)=:p"),
                {"p": norm}
            ).first()

            # ---------- Product NOT in inventory ----------
            if not row:
                # Try web price as a benchmark
                vendor_list = []
                try:
                    res = search_products.invoke({
                        "query": f"buy {product} UK price",
                        "filters": {"currency": "GBP", "max_results": 8}
                    })
                    if isinstance(res, dict):
                        vendor_list = res.get("candidates", []) or []
                except Exception as e:
                    print("[purchaseorder_approval] search_products error:", e)

                proposed_price = None
                if vendor_list:
                    best = next((c for c in vendor_list if isinstance(c.get("price"), (int, float))), None)
                    if best:
                        proposed_price = float(best["price"])
                        vendor_name = (best.get("vendor") or "N/A")[:200]
                        vendor_url = best.get("url") or "N/A"

                # Ask user to accept web price or provide one
                if proposed_price is not None:
                    decision = interrupt(
                        f"I couldn't find {product} in internal inventory.\n\n"
                        f"Best web price I found: £{proposed_price:.2f} at {vendor_name}.\n"
                        f"Use this price? Type 'yes' to accept or 'no' to enter a different price.\n"
                        f"Link: {vendor_url}"
                    )
                    if str(decision).strip().lower() == "yes":
                        unit_price = proposed_price
                    else:   
                        price_input = interrupt(
                            f"Please provide a unit price for {product} (number only):"
                        )
                        try:
                            unit_price = float(str(price_input).strip())
                        except Exception:
                            return {"message": f"Invalid unit price input: '{price_input}'. Buying declined.", "__end__": True}
                else:
                    price_input = interrupt(
                        f"The product {product} is not in inventory. Please provide a unit price (number only)?:"
                    )
                    try:
                        unit_price = float(str(price_input).strip())
                    except Exception:
                        return {"message": f"Invalid unit price input: '{price_input}'. Buying declined.", "__end__": True}

                # Insert new inventory, create PO, update stock
                conn.execute(
                    text("INSERT INTO procurement.Inventory (Product, Available, UnitPrice) VALUES (:Product, :available, :unit_price)"),
                    {"Product": norm, "available": quantity, "unit_price": unit_price}
                )
                conn.commit()

                Product = norm
                total = unit_price * quantity
                po_number, po_id = _insert_purchase_order(
                    eng, Product=Product, quantity=quantity, unit_price=unit_price,
                    vendor_name=vendor_name, vendor_url=vendor_url
                )
                _generate_po_pdf(
                    po_id=po_number, Product=Product, quantity=quantity, unit_price=unit_price,
                    total_price=total, vendor_name=vendor_name, vendor_url=vendor_url
                )

                conn.execute(
                    text("UPDATE procurement.Inventory SET Available = Available + :qty WHERE UPPER(Product) = :Product"),
                    {"qty": quantity, "Product": norm}
                )
                conn.commit()
                return {"message": f"New inventory created and order placed. PO: {po_number}", "__end__": True}

            # ---------- Product IN inventory ----------
            Product, available, unit_price = row.Product, int(row.Available), float(row.UnitPrice)

            # If you intend to prevent buying above availability, keep this (otherwise remove).
            if quantity > available:
                return {"message": f"Only {available} of {Product} available. Buying declined.", "__end__": True}

            total = unit_price * quantity

            # Approval rules
            decision = "no"
            if total <= 500:
                decision = "yes"
            elif total > 500 and total <= 5000:
                # Ask user with policy context
                decision = interrupt(
                    f"Approve buying {quantity} {Product} (stock: {available}) at £{unit_price:.2f} each "
                    f"(total £{total:.2f})? (yes/no)\n\nPolicy Summary:\n{policy_summary}"
                )
            else:
                return {"message": f"Order total £{total:.2f} exceeds Limit approval and requires finance approval.", "__end__": True}

            if str(decision).strip().lower() != "yes":
                return {"message": "Buying declined.", "__end__": True}

            # Proceed with PO; vendor fields are defined ("N/A") unless you add a vendor source here
            po_number, po_id = _insert_purchase_order(
                eng, Product=Product, quantity=quantity, unit_price=unit_price,
                vendor_name=vendor_name, vendor_url=vendor_url
            )
            _generate_po_pdf(
                po_id=po_number, Product=Product, quantity=quantity, unit_price=unit_price,
                total_price=total, vendor_name=vendor_name, vendor_url=vendor_url
            )

            with eng.connect() as conn2:
                conn2.execute(
                    text("UPDATE procurement.Inventory SET Available = Available + :qty WHERE UPPER(Product) = :Product"),
                    {"qty": quantity, "Product": Product.upper()}
                )
                conn2.commit()

            return {"message": f"Order placed, inventory updated, and PO generated. PO: {po_number}", "__end__": True}

    except Exception as e:
        return {"message": f"purchaseorder_approval error: {e}", "__end__": True}


# ---------- ReAct Agent ----------
if create_react_agent is None:
    raise ImportError("LangGraph ReAct agent not found. Upgrade langgraph: pip install -U langgraph")

tools = [purchaseorder_approval, policy_lookup, search_products]
llm = init_chat_model("azure_openai:gpt-35-turbo", azure_deployment="gpt-35-turbo")
memory = MemorySaver()
react_graph = create_react_agent(llm, tools=tools, checkpointer=memory)

# ---------- FastAPI ----------
app = FastAPI(title="PO Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    Question: Optional[str] = Field(None)
    decision: Optional[str] = Field(None)
    conversation_id: Optional[str] = Field(None)

@app.post("/agnt/chat")
def agnt_chat(req: ChatRequest):
    thread_id = req.conversation_id or f"buy_thread:{uuid4()}"
    config = RunnableConfig(configurable={"thread_id": thread_id})
    print(f"[agnt_chat] Received request: question='{req.Question}', decision='{req.decision}', thread_id='{thread_id}'")
    if req.decision:
        print(f"[agnt_chat] Resuming thread {thread_id} with decision: {req.decision}")
        state = react_graph.invoke(Command(resume=req.decision), config=config)
        return JSONResponse({"status": "DONE", "thread_id": thread_id, "message": state["messages"][-1].content})

    question = (req.Question or "").strip()
    if not question:
        return JSONResponse({"status":"ERROR","error":"Question is required"}, status_code=400)

    state = react_graph.invoke({"messages":[{"role":"user","content":question}]}, config=config)

    if "__interrupt__" in state:
        prompt = interrupt_text(state["__interrupt__"])
        return JSONResponse({"status":"APPROVAL_REQUIRED", "thread_id": thread_id, "prompt": prompt})

    return JSONResponse({"status":"DONE", "thread_id": thread_id, "message": state["messages"][-1].content})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("po_app_react_agentHITL:app", host="0.0.0.0", port=8000, reload=True)
