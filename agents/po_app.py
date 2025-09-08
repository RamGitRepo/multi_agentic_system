
from __future__ import annotations
import os
import urllib.parse
from typing import Dict, Optional
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import requests
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from azure.core.exceptions import ResourceNotFoundError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
import uvicorn

# LangGraph (agentic tool-calling)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState  # <-- IMPORTANT

# LangChain
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from langchain.chat_models import init_chat_model

# Azure OpenAI (embeddings & chat for refine tool)
from openai import AzureOpenAI

# Helpers
from helper.currency_helper import _extract_price_currency
from helper.po_blobstorage import generate_po_pdf, _get_blob_service_client, _abs_pdf_url
from helper.po_helpers import (
    _insert_purchase_order,
    _normalize_product,
    interrupt_text,
    _last_assistant_text,
)

# ------------------------- ENV / CLIENTS -------------------------
load_dotenv(find_dotenv(), override=True)


# DB
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DRIVER_PATH = "/opt/homebrew/opt/msodbcsql18/lib/libmsodbcsql.18.dylib"
PO_BLOB_CONTAINER = os.getenv("BLOB_CONTAINER")


ODBC_STR = (
    "Driver={{{driver_path}}};".format(driver_path=DRIVER_PATH) +
    f"Server=tcp:ramdemosql.database.windows.net,1433;"
    f"Database={DB_DATABASE};"
    f"Uid={DB_USER};"
    f"Pwd={DB_PASSWORD};"
    "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
)


# Azure Search + Azure OpenAI
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("INDEX_NAME") or "po-policy-index"

AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_ENGINE") or "text-embedding-ada-002"
# Chat LLM (tool-calling)
AZURE_CHAT_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME") or "gpt-4o"
llm = init_chat_model("azure_openai:gpt-4o", azure_deployment=AZURE_CHAT_DEPLOYMENT)

# Tavily
TAVILY_ENDPOINT = "https://api.tavily.com/search"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Timezone for “today”
LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/London"))

if not (AOAI_ENDPOINT and AOAI_KEY and AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY):
    raise ValueError("Missing Azure env vars (OPENAI endpoint/key or SEARCH endpoint/key).")

# Single AOAI client (embeddings + refine-query chat)
aoai_client = AzureOpenAI(
    api_key=AOAI_KEY,
    azure_endpoint=AOAI_ENDPOINT,
    api_version="2024-05-01-preview"
)


# ------------------------- DB Connection -------------------------
_engine = None
def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    db_url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(ODBC_STR)
    _engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=1800)
    return _engine



# ----------------------------- TOOLS -----------------------------


# ---------- Refine Query tool ----------
@tool
def refine_query(text: str) -> Dict:
    """
    Normalize a noisy user query into a clear, intent-focused form.
    Returns: {"refined": str}
    """
    try:
        prompt = f"""
You are a smart assistant that refines casual or vague user queries into clear, structured intent statements.
Rephrase the input for clarity and remove filler words.

Original Query: "{(text or '').strip()}"
Refined Intent:
""".strip()

        resp = aoai_client.chat.completions.create(
            model=AZURE_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a query refiner that restructures raw user questions into clean, intent-focused queries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=60,
        )
        content = (resp.choices[0].message.content or "").strip()
        refined = content if content else "No refined query generated."
        print(f"[refine_query] refined='{refined}'")
        return {"refined": refined}
    except Exception as e:
        fallback = (text or "").strip()
        print(f"[refine_query] ⚠️ error: {e}; fallback='{fallback}'")
        return {"refined": fallback or "purchase order request"}


# ---------- Policy lookup ----------
@tool
def policy_lookup(query: str) -> str:
    """policy guidelines/rules/restrictions; returns policy_summary."""
    emb = aoai_client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    query_vector = emb.data[0].embedding
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version=2024-07-01"
    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_KEY}
    payload = {
        "vectorQueries": [{"kind":"vector","vector":query_vector,"fields":"contentVector","k":5}],
        "select": "title,content"
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code != 200:
            return f"Policy lookup error: {resp.text}"
        data = resp.json()
    except Exception as e:
        return f"Policy lookup error: {e}"
    policy_summary = []
    for doc in data.get("value", []):
        policy_summary.append(f"- {doc.get('title','')}: {doc.get('content','')}")
    return "\n".join(policy_summary) if policy_summary else "No relevant policy found."


# ---------- Inventory check ----------
@tool
def check_availability(product: str) -> Dict:
    """Return availability + unit price from procurement.Inventory."""
    eng = _get_engine()
    norm = _normalize_product(product)
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT Product, Available, UnitPrice FROM procurement.Inventory WHERE UPPER(Product) = :p"),
            {"p": norm}
        ).first()
        if not row:
            row = conn.execute(
                text("""SELECT TOP 1 Product, Available, UnitPrice
                        FROM procurement.Inventory
                        WHERE UPPER(Product) LIKE :p
                        ORDER BY Product"""),
                {"p": norm + "%"}
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


# ---------- Web product search ----------
@tool
def search_products(product: str, filters: Optional[Dict] = None) -> Dict:
    """Search web for price to approve/order/create PO using product in the trusted Retailer, officail web sites for accurate price. Returns {candidates[], unit_price_hint, vendor_name, vendor_url}."""
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return {"error": "TAVILY_API_KEY missing/invalid", "candidates": [], "unit_price_hint": None, "vendor_name": None, "vendor_url": None}

    filters = filters or {}
    currency = (filters.get("currency") or "GBP").upper()
    max_results = int(filters.get("max_results") or 8)
    include_domains = filters.get("include_domains")

    TAVILY_ENDPOINT = "https://api.tavily.com/search"
    headers = {"Content-Type":"application/json","Accept":"application/json","Authorization":f"Bearer {api_key}"}
    payload = {
        "query": product, "search_depth": "advanced",
        "max_results": max(1, min(max_results, 10)),
        "include_answer": False, "include_images": False, "topic": "general",
    }
    if include_domains: payload["include_domains"] = include_domains

    try:
        session = requests.Session(); session.trust_env = False
        resp = session.post(TAVILY_ENDPOINT, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            return {"error": f"Tavily error {resp.status_code}: {resp.text}", "candidates": [], "unit_price_hint": None, "vendor_name": None, "vendor_url": None}
        data = resp.json()
    except Exception as e:
        return {"error": f"Tavily request failed: {e}", "candidates": [], "unit_price_hint": None, "vendor_name": None, "vendor_url": None}

    results = data.get("results", []) or []
    candidates = []
    best = None
    for r in results:
        url = r.get("url","") or ""
        title = r.get("title","") or ""
        content = r.get("content","") or ""
        domain = urllib.parse.urlparse(url).netloc
        price, cur = _extract_price_currency(f"{title} {content}", currency_hint=currency)
        cand = {"title": title[:200], "vendor": domain, "price": price, "currency": cur or currency, "url": url, "source": "tavily"}
        candidates.append(cand)
        if isinstance(price, (int, float)) and best is None:
            best = cand

    priced = [c for c in candidates if isinstance(c.get("price"), (int,float))]
    priced.sort(key=lambda x: x["price"])
    if priced:
        best = priced[0]
    
    
    return {
        "query": product,
        "candidates": (priced if priced else candidates)[:max_results],
        "unit_price_hint": (best.get("price") if best else None),
        "vendor_name": (best.get("vendor") if best else None),
        "vendor_url": (best.get("url") if best else None),
    }


# ---------- PO activity summary ----------
@tool
def po_activity_summary(date_str: Optional[str] = None) -> Dict:
    """
    Summarize Purchase Orders for a specific local date (defaults to 'today' in Europe/London).
    Returns: {date, total_orders, total_units, total_spend, top_items[], orders[]}
    """
    try:
        if date_str:
            y, m, d = map(int, date_str.split("-"))
            start_local = datetime(y, m, d, 0, 0, 0, tzinfo=LOCAL_TZ)
        else:
            now_local = datetime.now(LOCAL_TZ)
            start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_local = start_local + timedelta(days=1)
    except Exception:
        now_local = datetime.now(LOCAL_TZ)
        start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_local = start_local + timedelta(days=1)

    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)

    eng = _get_engine()
    table_candidates = [
        "procurement.PurchaseOrders",
        "procurement.PO",
        "dbo.PurchaseOrders",
        "PurchaseOrders",
    ]

    rows = []
    last_error = None
    with eng.connect() as conn:
        for tbl in table_candidates:
            try:
                q = text(f"""
                    SELECT PoNumber, Product, Quantity, UnitPrice, VendorName, VendorUrl, CreatedAt
                    FROM {tbl}
                    WHERE CreatedAt >= :start AND CreatedAt < :end
                    ORDER BY CreatedAt DESC
                """)
                res = conn.execute(q, {"start": start_utc, "end": end_utc}).fetchall()
                if res:
                    for r in res:
                        try:
                            rows.append({
                                "po":      r.PoNumber,
                                "product": r.Product,
                                "qty":     int(r.Quantity),
                                "unit":    float(r.UnitPrice),
                                "total":   float(r.Quantity) * float(r.UnitPrice),
                                "vendor":  getattr(r, "VendorName", None),
                                "url":     getattr(r, "VendorUrl", None),
                                "ts":      str(getattr(r, "CreatedAt", "")),
                            })
                        except Exception:
                            po, prod, qty, unit, vname, vurl, ts = r
                            rows.append({
                                "po": po, "product": prod, "qty": int(qty),
                                "unit": float(unit), "total": float(qty) * float(unit),
                                "vendor": vname, "url": vurl, "ts": str(ts),
                            })
                    break
                else:
                    rows = []
                    break
            except Exception as e:
                last_error = str(e)
                continue

    total_orders = len(rows)
    total_units = sum(r["qty"] for r in rows)
    total_spend = sum(r["total"] for r in rows)

    top_map: Dict[str, Dict[str, float]] = {}
    for r in rows:
        p = r["product"]
        t = top_map.setdefault(p, {"qty": 0, "spend": 0.0})
        t["qty"] += r["qty"]
        t["spend"] += r["total"]
    top_items = sorted(
        [{"product": k, "qty": int(v["qty"]), "spend": float(v["spend"])} for k, v in top_map.items()],
        key=lambda x: (x["qty"], x["spend"]), reverse=True
    )[:5]

    result = {
        "date": start_local.date().isoformat(),
        "total_orders": total_orders,
        "total_units": total_units,
        "total_spend": round(total_spend, 2),
        "top_items": top_items,
        "orders": rows,
    }
    if total_orders == 0 and last_error:
        result["note"] = f"No orders found and last table error: {last_error}"
    return result


# ---------- PO approval (HITL) ----------
@tool
def purchaseorder_approval(
    product: str,
    quantity: int,
    policy_summary: Optional[str] = None,
    unit_price_hint: Optional[float] = 100,
    vendor_name: Optional[str] = None,
    vendor_url: Optional[str] = None
) -> Dict:
    
    """Apporve and create a PO for the given product and quantity."""

    eng = _get_engine()
    norm = _normalize_product(product)
    vendor_name = (vendor_name or "N/A")[:200]
    vendor_url = vendor_url or "N/A"
    policy_summary = policy_summary or "Spending thresholds: Orders up to £500 total is auto-approved. Orders above £5,000 require finance Team approval and cannot be auto-approved"

    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT Product, Available, UnitPrice FROM procurement.Inventory WHERE UPPER(Product)=:p"),
            {"p": norm}
        ).first()

        if not row:
            if unit_price_hint is None:
                price_input = interrupt(f"'{product}' is not in inventory.\nPlease provide a unit price (number only)?")
                unit_price = float(str(price_input).strip())
            else:
                unit_price = float(unit_price_hint)


            Product = norm; 
            total = unit_price * quantity
            available = 0

            if total <= 500:
                approve = "yes"
            elif total <= 5000 and quantity <= 50:
                approve = interrupt(
                    f"Approve buying {quantity} {Product} (stock: {available}) and Proposed Price is £{unit_price:.2f} each "
                    f"(total £{total:.2f})? (Yes/No)\n\nPolicy Summary:\n{policy_summary}"
                )
            else:
                return {"message": f"Order total/units £{total:.2f} and {quantity}  and Proposed Price is £{unit_price:.2f}  exceeds limit and requires finance approval.", "__end__": True}
            
            if str(approve).strip().lower() != "yes":
                return {"message": "Buying declined.", "__end__": True}
            
            # insert inventory + PO
            conn.execute(
                text("INSERT INTO procurement.Inventory (Product, Available, UnitPrice) VALUES (:Product, :available, :unit_price)"),
                {"Product": norm, "available": quantity, "unit_price": unit_price}
            ); conn.commit()
            
            po_number, _ = _insert_purchase_order(
                eng, Product=Product, quantity=quantity, unit_price=unit_price,
                vendor_name=vendor_name, vendor_url=vendor_url
            )

            pdf_info = generate_po_pdf(
                po_id=po_number,
                Product=Product,
                quantity=quantity,
                unit_price=unit_price,
                total_price=total,
                vendor_name=vendor_name,
                vendor_url=vendor_url,
                )
            
            pdf_url = pdf_info["api_url"]     
            p_url = _abs_pdf_url(pdf_url) 
            #return {"message": f"Order placed and inventory seeded. PO Number: {po_number}", "__end__": True}
            return {"message": f"Order placed, inventory seeded. PO generated. PO Number: {po_number}","View/Download PDF": p_url ,"__end__": True}


        # in inventory
        Product, available, unit_price = row.Product, int(row.Available), float(row.UnitPrice)
        total = unit_price * quantity

        if total <= 500:
            approve = "yes"
        elif total <= 5000 and quantity <= 50:
            approve = interrupt(
                f"Approve buying {quantity} {Product} (stock: {available}) at £{unit_price:.2f} each "
                f"(total £{total:.2f})? (Yes/No)\n\nPolicy Summary:\n{policy_summary}"
            )
        else:
            return {"message": f"Order total/units £{total:.2f} and {quantity} exceeds limit and requires finance approval.", "__end__": True}

        if str(approve).strip().lower() != "yes":
            return {"message": "Buying declined.", "__end__": True}

        po_number, _ = _insert_purchase_order(
            eng, Product=Product, quantity=quantity, unit_price=unit_price,
            vendor_name=vendor_name, vendor_url=vendor_url
        )

        pdf_info = generate_po_pdf(
                po_id=po_number,
                Product=Product,
                quantity=quantity,
                unit_price=unit_price,
                total_price=total,
                vendor_name=vendor_name,
                vendor_url=vendor_url,
                )

       
        with eng.connect() as conn2:
            conn2.execute(text("UPDATE procurement.Inventory SET Available = Available + :qty WHERE UPPER(Product) = :Product"),
                          {"qty": quantity, "Product": Product.upper()})
            conn2.commit()
         
 
        pdf_url = pdf_info["api_url"]
        p_url = _abs_pdf_url(pdf_url) 
        #return {"message": f"Order placed, inventory updated, and PO generated. PO: {po_number}", "__end__": True}
        return {"message": f"Order placed, inventory updated, and PO generated. PO Number: {po_number}", "View/Download PDF": p_url,"__end__": True}


# ---------------------- AGENTIC TOOL-CALLING GRAPH ----------------------
# Register tools (refine_query first so the LLM tends to call it early)
TOOLS = [refine_query, policy_lookup, check_availability, search_products, po_activity_summary, purchaseorder_approval]

AGENTIC_SYSTEM = (
    "You are a procurement assistant. Use tools to stay factual.\n"
    "Workflow guidance:\n"
    "1) If the user's query is long, noisy, or unclear, FIRST call refine_query(text) and use its 'refined' output as the canonical query for subsequent tools.\n"
    "2) For policy/rules/thresholds and policy_summary, call policy_lookup.\n"
    "3) For stock/price, call check_availability(product).\n"
    "4) If not in inventory, call search_products(product) to get unit_price_hint/vendor fields, then pass them to purchaseorder_approval.\n"
    "5) Use purchaseorder_approval for creating POs (pass policy_summary and any vendor/unit price hints).\n"
    "6) Use po_activity_summary for daily metrics.\n"
    "Prefer tool facts; answer concisely. If refine_query returns a better phrasing, prefer that wording in your final answer and tool arguments."
)

def agent_node(state: MessagesState) -> dict:
    msgs = state.get("messages", [])
    #ßprint("[agent_node] Invoking LLM with messages:", msgs)
    ai = llm.bind_tools(TOOLS).invoke(msgs)
    print("[agent_node] tool_calls:", getattr(ai, "tool_calls", None))
    return {"messages": [ai]}   # append assistant message

tool_node = ToolNode(TOOLS)

def msg_router(state: MessagesState) -> str:
    msgs = state.get("messages", [])
    if msgs:
        last = msgs[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        if isinstance(last, dict) and last.get("tool_calls"):
            return "tools"
    return "finalize"

def finalize_node(state: MessagesState) -> MessagesState:
    return state

# Build graph
po_agent_wf = StateGraph(MessagesState)
po_agent_wf.add_node("agent", agent_node)
po_agent_wf.add_node("tools", tool_node)
po_agent_wf.add_node("finalize", finalize_node)
po_agent_wf.set_entry_point("agent")
po_agent_wf.add_conditional_edges("agent", msg_router, {
    "tools": "tools",
    "finalize": "finalize",
})
po_agent_wf.add_edge("tools", "agent")

agent_memory = MemorySaver()
agent_graph = po_agent_wf.compile(checkpointer=agent_memory)



# ----------------------------- FASTAPI -----------------------------
app = FastAPI(title="PO_Agent_API", version="3.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
  Question: Optional[str] = Field(None)
  decision: Optional[str] = Field(None)
  conversation_id: Optional[str] = Field(None)


@app.post("/agnt/chat")
def agnt_agent(req: ChatRequest):
    thread_id = req.conversation_id or f"agent_thread:{uuid4()}"
    config = RunnableConfig(configurable={"thread_id": thread_id})

    # Resume HITL interrupts (e.g., approvals / manual price)
    if req.decision:
        state = agent_graph.invoke(Command(resume=str(req.decision).strip().lower()), config=config)
        return JSONResponse({"status":"DONE","thread_id":thread_id,"message":_last_assistant_text(state.get("messages", []))})

    question = (req.Question or "").strip()
    if not question:
        return JSONResponse({"status":"ERROR","error":"Question is required"}, status_code=400)

    # Initial messages
    msgs = [{"role": "user", "content": question}]
    if not req.conversation_id:
        msgs = [{"role": "system", "content": AGENTIC_SYSTEM}] + msgs

    # Run agentic loop
    state = agent_graph.invoke({"messages": msgs}, config=config)  # type: ignore

    if "__interrupt__" in state:
        prompt = interrupt_text(state["__interrupt__"])
        return JSONResponse({"status":"APPROVAL_REQUIRED", "thread_id": thread_id, "prompt": prompt})

    return JSONResponse({"status":"DONE","thread_id":thread_id,"message":_last_assistant_text(state.get("messages", []))})


@app.get("/po/{po_id}/pdf")
def download_po_pdf(po_id: str, download: bool = False):
    """
    Stream the PO document from Azure Blob Storage.
    Tries PDF first, then TXT fallback if PDF doesn't exist.
    Use ?download=true to force 'attachment' download; otherwise renders inline.
    """
    svc = _get_blob_service_client()
    container = svc.get_container_client(PO_BLOB_CONTAINER) # type: ignore
    print(f"Fetching PO document for id '{po_id}' from blob container '{PO_BLOB_CONTAINER}'")

    # Our generate_po_pdf stores as: purchaseorders/PO_<po_id>/PO_<po_id>.pdf (or .txt)
    candidates = [
        (f"{po_id}/{po_id}.pdf", "application/pdf"),
        (f"{po_id}/{po_id}.txt", "text/plain; charset=utf-8"),
    ]

    last_err = None
    for blob_name, default_type in candidates:
        try:
            bc = container.get_blob_client(blob_name)
            props = bc.get_blob_properties()  # will raise if not found
            content_type = (props.content_settings.content_type or default_type)
            # Stream the blob in chunks to the client
            downloader = bc.download_blob()
            filename = blob_name.split("/")[-1]
            disp = "attachment" if download else "inline"
            headers = {
                "Content-Disposition": f'{disp}; filename="{filename}"'
            }
            return StreamingResponse(downloader.chunks(), media_type=content_type, headers=headers)
        except ResourceNotFoundError as e:
            last_err = e
            continue
        except Exception as e:
            # Any other unexpected error
            raise HTTPException(status_code=500, detail=f"Error streaming blob: {e}")

    # If neither PDF nor TXT was found
    raise HTTPException(status_code=404, detail=f"PO document not found for id '{po_id}'.")


if __name__ == "__main__":
    uvicorn.run("po_app:app", host="0.0.0.0", port=8000, reload=True)
