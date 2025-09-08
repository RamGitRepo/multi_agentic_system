# po_app_react_agent_rag.py
from __future__ import annotations

import os
import re
import json
import urllib.parse
from typing import Annotated, Dict, Optional, TypedDict
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

# LangGraph (deterministic graph)
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# LangChain
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from langchain.chat_models import init_chat_model

# Azure OpenAI (embeddings client for policy lookup)
from openai import AzureOpenAI

# Your helpers
from helper.currency_helper import _extract_price_currency
from helper.po_helpers import _insert_purchase_order, _generate_po_pdf


# ------------------------- ENV / CLIENTS -------------------------
load_dotenv(find_dotenv(), override=True)

# DB
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

# Azure Search + Azure OpenAI
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX") or "po-policy-index"

AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_KEY = os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL") or "text-embedding-ada-002"

if not (AOAI_ENDPOINT and AOAI_KEY and AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY):
    raise ValueError("Missing Azure env vars (OPENAI endpoint/key or SEARCH endpoint/key).")

aoai_client = AzureOpenAI(api_key=AOAI_KEY, azure_endpoint=AOAI_ENDPOINT, api_version="2024-05-01-preview")

# Chat LLM (for extraction + final response)
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or "gpt-35-turbo"
llm = init_chat_model("azure_openai:gpt-35-turbo", azure_deployment=AZURE_CHAT_DEPLOYMENT)

# Tavily
TAVILY_ENDPOINT = "https://api.tavily.com/search"
TAVILY_API_KEY = (os.getenv("TAVILY_API_KEY") or "").strip()

# Timezone for “today”
LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/London"))


# ----------------------------- UTILS -----------------------------
def _normalize_product(name: str) -> str:
    n = re.sub(r"\b(Product|laptop|pc|pcs|computer)s?\b$", "", name.strip(), flags=re.I)
    return n.strip().upper()

def interrupt_text(obj) -> str:
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

def _role_and_content(msg):
    if isinstance(msg, dict):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "human": role = "user"
        if role == "ai": role = "assistant"
        return role, content
    msg_type = getattr(msg, "type", None)
    content = getattr(msg, "content", "")
    if msg_type == "human": return "user", content
    if msg_type == "ai": return "assistant", content
    return msg_type, content

def _last_assistant_text(messages) -> str:
    for m in reversed(messages or []):
        role, content = _role_and_content(m)
        if role == "assistant":
            return content
    if not messages: return ""
    m = messages[-1]
    return getattr(m, "content", m.get("content", "")) if isinstance(m, dict) else getattr(m, "content", "")


# ----------------------------- TOOLS -----------------------------
@tool
def policy_lookup(query: str) -> str:
    """product policy guidelines; returns short bulleted text."""
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
    items = []
    for doc in data.get("value", []):
        items.append(f"- {doc.get('title','')}: {doc.get('content','')}")
    return "\n".join(items) if items else "No relevant policy found."

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
            Product = row.Product; available = int(row.Available); unit_price = float(row.UnitPrice)
        except Exception:
            Product, available, unit_price = row[0], int(row[1]), float(row[2])
        return {"product": Product, "available": available, "unit_price": unit_price}
    return {"product": norm, "available": 0, "unit_price": 0.0}

@tool
def search_products(query: str, filters: Optional[Dict] = None) -> Dict:
    """Web price via Tavily. Returns normalized candidates."""
    api_key = TAVILY_API_KEY
    if not api_key or not api_key.startswith("tvly-"):
        return {"error": "TAVILY_API_KEY missing/invalid", "candidates": []}

    filters = filters or {}
    currency = (filters.get("currency") or "GBP").upper()
    max_results = int(filters.get("max_results") or 8)
    include_domains = filters.get("include_domains")

    headers = {"Content-Type":"application/json","Accept":"application/json","Authorization":f"Bearer {api_key}"}
    payload = {
        "query": query, "search_depth": "advanced",
        "max_results": max(1, min(max_results, 10)),
        "include_answer": False, "include_images": False, "topic": "general",
    }
    if include_domains: payload["include_domains"] = include_domains

    try:
        session = requests.Session(); session.trust_env = False
        resp = session.post(TAVILY_ENDPOINT, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            return {"error": f"Tavily error {resp.status_code}: {resp.text}", "candidates": []}
        data = resp.json()
    except Exception as e:
        return {"error": f"Tavily request failed: {e}", "candidates": []}

    results = data.get("results", []) or []
    candidates = []
    for r in results:
        url = r.get("url","") or ""; title = r.get("title","") or ""; content = r.get("content","") or ""
        domain = urllib.parse.urlparse(url).netloc
        price, cur = _extract_price_currency(f"{title} {content}", currency_hint=currency)
        candidates.append({"title": title[:200], "vendor": domain, "price": price, "currency": cur or currency, "url": url, "source": "tavily"})
    priced = [c for c in candidates if isinstance(c.get("price"), (int,float))]
    priced.sort(key=lambda x: x["price"])
    return {"query": query, "candidates": (priced if priced else candidates)[:max_results]}

# ---------------- NEW: Today's PO activity summary tool ----------------
@tool
def po_activity_summary(date_str: Optional[str] = None) -> Dict:
    """
    Summarize Purchase Orders for a specific local date (defaults to 'today' in Europe/London).
    Returns: {date, total_orders, total_units, total_spend, top_items[], orders[]}
    """
    # Compute local day window
    try:
        if date_str:
            # Accept YYYY-MM-DD
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

    # Candidate table names (try in order)
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
                            # tuple-style
                            po, prod, qty, unit, vname, vurl, ts = r
                            rows.append({
                                "po": po, "product": prod, "qty": int(qty),
                                "unit": float(unit), "total": float(qty) * float(unit),
                                "vendor": vname, "url": vurl, "ts": str(ts),
                            })
                    break
                else:
                    # still break; empty day is valid
                    rows = []
                    break
            except Exception as e:
                last_error = str(e)
                continue

    total_orders = len(rows)
    total_units = sum(r["qty"] for r in rows)
    total_spend = sum(r["total"] for r in rows)

    # Top items by units then spend
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
# -----------------------------------------------------------------------


@tool
def purchaseorder_approval(
    product: str,
    quantity: int,
    policy_summary: Optional[str] = None,
    unit_price_hint: Optional[float] = None,
    vendor_name: Optional[str] = None,
    vendor_url: Optional[str] = None
) -> Dict:
    """Final approval + PO creation. Assumes earlier steps filled hints."""
    eng = _get_engine()
    norm = _normalize_product(product)
    vendor_name = (vendor_name or "N/A")[:200]
    vendor_url = vendor_url or "N/A"
    policy_summary = policy_summary or "(No policy summary provided)"

    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT Product, Available, UnitPrice FROM procurement.Inventory WHERE UPPER(Product)=:p"),
            {"p": norm}
        ).first()

        if not row:
            # need price (hint or HITL)
            if unit_price_hint is None:
                price_input = interrupt(f"'{product}' is not in inventory.\nPlease provide a unit price (number only):")
                unit_price = float(str(price_input).strip())
            else:
                unit_price = float(unit_price_hint)

            conn.execute(
                text("INSERT INTO procurement.Inventory (Product, Available, UnitPrice) VALUES (:Product, :available, :unit_price)"),
                {"Product": norm, "available": quantity, "unit_price": unit_price}
            ); conn.commit()

            Product = norm; total = unit_price * quantity

            if total <= 5000:
                approve = "yes"
            else:
                approve = interrupt(
                    f"Approve buying {quantity} {Product} at £{unit_price:.2f} each (total £{total:.2f})? (Yes/No)\n\nPolicy Summary:\n{policy_summary}"
                )
            if str(approve).strip().lower() != "yes":
                return {"message": "Buying declined.", "__end__": True}

            po_number, _ = _insert_purchase_order(
                eng, Product=Product, quantity=quantity, unit_price=unit_price,
                vendor_name=vendor_name, vendor_url=vendor_url
            )
            _generate_po_pdf(po_id=po_number, Product=Product, quantity=quantity, unit_price=unit_price,
                             total_price=total, vendor_name=vendor_name, vendor_url=vendor_url)

            conn.execute(text("UPDATE procurement.Inventory SET Available = Available + :qty WHERE UPPER(Product) = :Product"),
                         {"qty": quantity, "Product": norm}); conn.commit()
            return {"message": f"Order placed and inventory seeded. PO Number: {po_number}", "__end__": True}

        # in inventory
        Product, available, unit_price = row.Product, int(row.Available), float(row.UnitPrice)
        if quantity > available:
            return {"message": f"Only {available} of {Product} available. Buying declined.", "__end__": True}

        total = unit_price * quantity
        if total <= 500:
            approve = "yes"
        elif total <= 5000:
            approve = interrupt(
                f"Approve buying {quantity} {Product} (stock: {available}) at £{unit_price:.2f} each "
                f"(total £{total:.2f})? (Yes/No)\n\nPolicy Summary:\n{policy_summary}"
            )
        else:
            return {"message": f"Order total £{total:.2f} exceeds limit and requires finance approval.", "__end__": True}

        if str(approve).strip().lower() != "yes":
            return {"message": "Buying declined.", "__end__": True}

        po_number, _ = _insert_purchase_order(
            eng, Product=Product, quantity=quantity, unit_price=unit_price,
            vendor_name=vendor_name, vendor_url=vendor_url
        )
        _generate_po_pdf(po_id=po_number, Product=Product, quantity=quantity, unit_price=unit_price,
                         total_price=total, vendor_name=vendor_name, vendor_url=vendor_url)

        with eng.connect() as conn2:
            conn2.execute(text("UPDATE procurement.Inventory SET Available = Available + :qty WHERE UPPER(Product) = :Product"),
                          {"qty": quantity, "Product": Product.upper()})
            conn2.commit()

        return {"message": f"Order placed, inventory updated, and PO generated. PO: {po_number}", "__end__": True}


# ---------------------------- GRAPH STATE ----------------------------
class POState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    # intents: now includes inventory + summary
    intent: Optional[str]            # "po" | "policy" | "inventory" | "summary"
    policy_query: Optional[str]      # raw policy question
    # existing
    product: Optional[str]
    quantity: Optional[int]
    policy_summary: Optional[str]
    inventory: Optional[Dict]
    web_candidates: Optional[list]
    unit_price_hint: Optional[float]
    vendor_name: Optional[str]
    vendor_url: Optional[str]
    summary_data: Optional[Dict]
    final_message: Optional[str]

def _parse_product_qty(text: str) -> Dict[str, Optional[str]]:
    s = (text or "").strip()
    m = re.search(r"(?i)\bcreate(?:\s+a)?\s+(?:purchase\s+order|po)\s+for\s+(\d+)\s+(.+)", s)
    if m: return {"qty": m.group(1), "product": m.group(2).strip()}
    m = re.search(r"(?i)\b(buy|order|need)\s+(\d+)\s+(.+)", s)
    if m: return {"qty": m.group(2), "product": m.group(3).strip()}
    m = re.search(r"(?i)\b(\d+)\s+(.+)", s)
    if m: return {"qty": m.group(1), "product": m.group(2).strip()}
    return {"qty": None, "product": None}

# ----------------------------- NODES -----------------------------
def node_extract_llm(state: POState) -> POState:
    msgs = state.get("messages", []) or []
    user_text = ""
    for msg in reversed(msgs):
        role, content = _role_and_content(msg)
        if role in ("user","human"):
            user_text = content; break

    system = (
        "Classify PO chat intent and extract slots. Return ONLY JSON with keys:\n"
        '  intent: "po" | "policy" | "inventory" | "summary"\n'
        "  product: string|null\n"
        "  quantity: int|null\n"
        "  policy_query: string|null\n"
        "Rules:\n"
        '- Use "inventory" for stock/availability questions (e.g., \"how many in stock\", \"availability\").\n'
        "- Use \"summary\" for daily/period activity reports (e.g., 'summarize today's PO activity', 'how many POs today', 'total spend today').\n"
        '- Use \"policy\" only for rules/thresholds/approvals.\n'
        '- Use \"po\" when the user wants to buy/order/create a PO.\n'
        "Do not invent values."
    )
    prompt = f"User message: {user_text}\nReturn JSON now."

    ai = llm.invoke([{"role":"system","content":system},{"role":"user","content":prompt}])
    content = getattr(ai, "content", "")

    intent = None
    product = None
    qty = None
    policy_query = None

    try:
        data = json.loads(content)
        intent = (data.get("intent") or "").lower() or None
        product = data.get("product") or None
        qty = data.get("quantity")
        policy_query = data.get("policy_query") or None
        if isinstance(qty, str) and qty.isdigit():
            qty = int(qty)
    except Exception:
        pass

    # Fallback heuristics
    s = (user_text or "").strip()
    if not intent:
        if re.search(r"(?i)\b(stock|availability|available|in\s+stock|how\s+many\s+(left|available))\b", s):
            intent = "inventory"
        elif re.search(r"(?i)\b(summary|summarize|report|activity|stats|metrics)\b", s) or re.search(r"(?i)\bhow\s+many\s+po", s):
            intent = "summary"
        elif re.search(r"(?i)\b(approval|policy|rules|threshold|finance)\b", s):
            intent = "policy"
        else:
            intent = "po"

    # Extract product if missing for inventory/po
    if intent in ("inventory","po") and not product:
        m = re.search(r"(?i)(?:stock|availability|available).*?(?:in|for|of)\s+(.+)$", s)
        if m:
            product = m.group(1).strip()
        else:
            parsed = _parse_product_qty(s)
            product = parsed["product"] or s

    # For pure policy
    if intent == "policy":
        return {**state, "intent": "policy", "policy_query": policy_query or s}

    # For summary: no product/qty needed
    if intent == "summary":
        return {**state, "intent": "summary"}

    # For inventory: no quantity needed
    if intent == "inventory":
        if not product:
            ans = interrupt("Which product should I check stock for?")
            product = str(ans).strip()
        return {**state, "intent": "inventory", "product": product}

    # Otherwise PO flow: collect missing product/qty via HITL
    if not product:
        ans = interrupt("What product do you want to buy?")
        product = str(ans).strip()

    if qty is None:
        ans = interrupt("How many units do you need?")
        try:
            qty = int(str(ans).strip())
        except Exception:
            ans2 = interrupt("Quantity must be an integer. Please enter quantity:")
            qty = int(str(ans2).strip())

    return {**state, "intent": "po", "product": product, "quantity": qty}  # type: ignore

def node_policy(state: POState) -> POState:
    intent = (state.get("intent") or "").lower()
    if intent == "policy":
        q = state.get("policy_query") or "purchase order approval rules"
        policy = policy_lookup.invoke(q)
        return {**state, "policy_summary": policy}
    elif intent == "po":
        q = f"purchase order approval rules for {state.get('product', 'unknown product')} quantity {state.get('quantity', 'unknown quantity')}"
        policy = policy_lookup.invoke(q)
        return {**state, "policy_summary": policy}
    else:
        # inventory / summary: skip policy lookup
        return {**state, "policy_summary": ""}

def node_inventory(state: POState) -> POState:
    product = state.get("product")
    if not product:
        raise ValueError("Product is required but not provided in the state.")
    inv = check_availability.invoke(product)
    return {**state, "inventory": inv}

def node_search_if_needed(state: POState) -> POState:
    inv = state.get("inventory") or {}
    available = int(inv.get("available", 0)) if inv else 0
    candidates = []; unit_price_hint = None; vendor_name = None; vendor_url = None
    if available <= 0:
        res = search_products.invoke({
            "query": f"buy {state.get('product', '')} UK price",
            "filters": {"currency": "GBP", "max_results": 8}
        })
        if isinstance(res, dict):
            candidates = res.get("candidates") or []
            best = next((c for c in candidates if isinstance(c.get("price"), (int,float))), None)
            if best:
                unit_price_hint = float(best["price"])
                vendor_name = (best.get("vendor") or "")[:200]
                vendor_url = best.get("url") or None
    return {**state, "web_candidates": candidates, "unit_price_hint": unit_price_hint,
            "vendor_name": vendor_name, "vendor_url": vendor_url}

def node_approval(state: POState) -> POState:
    result = purchaseorder_approval.invoke({
        "product": state.get("product"),
        "quantity": state.get("quantity"),
        "policy_summary": state.get("policy_summary"),
        "unit_price_hint": state.get("unit_price_hint"),
        "vendor_name": state.get("vendor_name"),
        "vendor_url": state.get("vendor_url"),
    })
    msg = result.get("message") if isinstance(result, dict) else str(result)
    return {**state, "final_message": msg}

def node_policy_answer(state: POState) -> POState:
    question = state.get("policy_query") or ""
    policy = state.get("policy_summary") or "No relevant policy found."
    system = "You are a concise procurement policy assistant. Keep answers short and precise."
    user = f"Q: {question}\nRelevant policy notes:\n{policy}\n\nWrite a clear answer."
    ai = llm.invoke([{"role":"system","content":system},{"role":"user","content":user}])
    final = getattr(ai, "content", policy) or policy
    return {**state, "final_message": final}

# NEW: inventory-only answer node
def node_inventory_answer(state: POState) -> POState:
    inv = state.get("inventory") or {}
    product = inv.get("product") or state.get("product") or "requested item"
    available = inv.get("available", 0)
    unit_price = inv.get("unit_price", None)

    if unit_price is not None:
        msg = f"Stock for {product}: {int(available)} unit(s) available at £{float(unit_price):.2f} each."
    else:
        msg = f"Stock for {product}: {int(available)} unit(s) available."

    return {**state, "final_message": msg}

# NEW: summary node
def node_summary(state: POState) -> POState:
    # optional date parsing (e.g., "summary for 2025-08-29"); for now always "today"
    data = po_activity_summary.invoke({})
    # Store raw data in state (useful for UI)
    state["summary_data"] = data

    # Compose a concise human answer
    date = data.get("date", "")
    n = int(data.get("total_orders", 0))
    units = int(data.get("total_units", 0))
    spend = float(data.get("total_spend", 0.0))
    tops = data.get("top_items", []) or []

    lines = [f"**PO Activity for {date}**",
             f"- Orders: {n}",
             f"- Units: {units}",
             f"- Total spend: £{spend:,.2f}"]

    if tops:
        lines.append("- Top items:")
        for t in tops[:3]:
            lines.append(f"  • {t.get('product')}: {int(t.get('qty',0))} unit(s), £{float(t.get('spend',0.0)):,.2f}")

    # Include a quick latest order if any
    orders = data.get("orders", [])
    if orders:
        last = orders[0]
        lines.append(f"- Latest: PO {last.get('po')} — {last.get('product')} x{last.get('qty')} at £{float(last.get('unit',0)):.2f} ({last.get('vendor')})")

    return {**state, "final_message": "\n".join(lines)}

def node_respond_llm(state: POState) -> POState:
    raw = state.get("final_message") or ""
    system = "You are a concise PO assistant. Keep responses brief and actionable."
    user = f"Rewrite this for the user, preserve all facts/PO numbers:\n\n{raw}"
    ai = llm.invoke([{"role":"system","content":system},{"role":"user","content":user}])
    nice = getattr(ai, "content", raw) or raw
    return {**state, "final_message": nice}

def node_respond(state: POState) -> POState:
    msgs = state.get("messages", [])
    msgs.append(AIMessage(content=state.get("final_message") or ""))
    state["messages"] = msgs
    return state

# ---------------------- WIRE GRAPH (ORDER + BRANCH) ----------------------
def route_after_policy(state: POState) -> str:
    i = (state.get("intent") or "").lower()
    if i == "policy": return "policy_answer"
    if i == "summary": return "summary"
    return "inventory"  # default path for po/inventory

def route_after_inventory(state: POState) -> str:
    return "inventory_answer" if (state.get("intent") or "").lower() == "inventory" else "websearch"

workflow = StateGraph(POState)
workflow.add_node("extract",   node_extract_llm)
workflow.add_node("policy",    node_policy)
workflow.add_node("policy_answer", node_policy_answer)
workflow.add_node("inventory", node_inventory)
workflow.add_node("inventory_answer", node_inventory_answer)
workflow.add_node("summary",   node_summary)              # NEW
workflow.add_node("websearch", node_search_if_needed)
workflow.add_node("approval",  node_approval)
workflow.add_node("respond_llm", node_respond_llm)
workflow.add_node("respond",   node_respond)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "policy")
workflow.add_conditional_edges("policy", route_after_policy, {
    "policy_answer": "policy_answer",
    "summary": "summary",               # NEW
    "inventory": "inventory",
})
workflow.add_conditional_edges("inventory", route_after_inventory, {
    "inventory_answer": "inventory_answer",
    "websearch": "websearch",
})
workflow.add_edge("policy_answer", "respond_llm")
workflow.add_edge("summary", "respond_llm")              # NEW
workflow.add_edge("inventory_answer", "respond_llm")
workflow.add_edge("websearch", "approval")
workflow.add_edge("approval", "respond_llm")
workflow.add_edge("respond_llm", "respond")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


# ----------------------------- FASTAPI -----------------------------
app = FastAPI(title="PO Agent API (Deterministic Graph + Policy/Inventory/Summary)", version="1.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    Question: Optional[str] = Field(None)
    decision: Optional[str] = Field(None)
    conversation_id: Optional[str] = Field(None)

@app.post("/agnt/chat")
def agnt_chat(req: ChatRequest):
    thread_id = req.conversation_id or f"buy_thread:{uuid4()}"
    config = RunnableConfig(configurable={"thread_id": thread_id})

    # Resume HITL interrupts
    if req.decision:
        state = graph.invoke(Command(resume=str(req.decision).strip().lower()), config=config)
        return JSONResponse({"status": "DONE", "thread_id": thread_id, "message": _last_assistant_text(state["messages"])})

    question = (req.Question or "").strip()
    if not question:
        return JSONResponse({"status":"ERROR","error":"Question is required"}, status_code=400)

    # Start run
    state = graph.invoke({"messages":[{"role":"user","content":question}]}, config=config)  # type: ignore

    # If any node raised interrupt(...)
    if "__interrupt__" in state:
        prompt = interrupt_text(state["__interrupt__"])
        return JSONResponse({"status":"APPROVAL_REQUIRED", "thread_id": thread_id, "prompt": prompt})

    return JSONResponse({"status":"DONE", "thread_id": thread_id, "message": _last_assistant_text(state["messages"])})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("po_app_HITL:app", host="0.0.0.0", port=8000, reload=True)
