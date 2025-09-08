<img width="836" height="667" alt="Langgraphagent Architecture" src="https://github.com/user-attachments/assets/d88b0048-a9f6-43cf-bd25-3dcca3845fdf" />

###  💼 `PO Assistant AI – Agentic Procurement System`

This project is an intelligent Purchase Order (PO) Assistant built using FastAPI, LangGraph, and Azure OpenAI, with integrations into Azure Cognitive Search, SQL Database, Azure Blob Storage, and external retail search APIs.

It automates procurement workflows — from product price lookups and policy validation to PO approvals and PDF generation — all with human-in-the-loop (HITL) support for finance compliance.

--- 

### 🔧 `Features`

✅ Agentic workflow orchestration with LangGraph + LangChain tools

🔍 Azure Cognitive Search for procurement policy retrieval

🗂 SQL Server (Azure SQL) for inventory + PO records

📦 Azure Blob Storage for PO PDF storage and retrieval

🌐 Web search integration (Tavily API) for trusted product/vendor pricing

🧾 PO approval flow with HITL interrupts for manager/finance sign-off

🔁 End-to-end automation from query → price lookup → policy validation → PO generation

### 🧠 `Tool Architecture`
### 📄 `refine_query`

Uses Azure OpenAI to normalize noisy procurement requests.

Example:

Input: “can u get me price for dell monitor plz??”

Output: “Find Dell Monitor price for purchase order creation.”

### 📄 `policy_lookup`

Queries Azure Cognitive Search with embeddings to retrieve policy guidelines.

Returns procurement rules such as:

Spending thresholds

Approval limits

Preferred vendors

### 📄 `check_availability`

Queries Azure SQL Inventory table.

Returns:

Current stock availability

Unit price

### 📄 `search_products`

Uses Tavily Search API to fetch product/vendor info from official retailer sites.

Returns structured JSON:

{
  "candidates": ["Dell Monitor - £200"],
  "unit_price_hint": 200,
  "vendor_name": "Official Retailer",
  "vendor_url": "https://retailer.com/product"
}

### 📄 `po_activity_summary`

Summarizes daily procurement activity.

Returns:

Date

Total orders

Total spend

Top items purchased

Order details

### 📄 `purchaseorder_approval`

Handles approval + PO creation.

Auto-approves low-value orders (< £500).

Interrupts for manager approval when needed (≤ £5,000).

Sends to finance team escalation for high-value orders.

Generates a PO PDF and stores it in Azure Blob Storage.

### 🔁 `Orchestrated PO Flow`

User Input
↓
Agent receives input
↓
➀ refine_query() → Normalize procurement request
↓
➁ policy_lookup() → Validate procurement rules
↓
➂ check_availability() → Look up product in SQL inventory
↓
➃ search_products() → Fetch vendor/price from trusted sources (if not in inventory)
↓
➄ purchaseorder_approval() → Create PO, update inventory, generate PDF
↓
Final Response → Order confirmation + PO PDF link

### 🤖 `Assistant API (FastAPI)`

Built with FastAPI for REST endpoints.

Key routes:

POST /agnt/chat → Conversational procurement assistant

GET /po/{po_id}/pdf → Stream/download PO document from Azure Blob Storage

CORS-enabled for frontend integration (e.g., React UI).

### ✨ `Features`

🔹 End-to-end PO automation with HITL decision points

🔹 Azure SQL integration for inventory + order persistence

🔹 PO document streaming (PDF/TXT fallback) from Azure Blob

🔹 UUID-based conversation/session tracking

🔹 Configurable approval policies via Azure Cognitive Search

### 📈 `Observability & Monitoring`

Logs tool calls and responses for traceability

Tracks HITL approval interrupts for auditing

Monitors order thresholds and compliance triggers

Structured logging for integration with Azure Monitor / App Insights

### 📌 `Coming Soon`

Integration with ERP systems (SAP, Oracle)

Vendor comparison dashboards (multi-supplier price intelligence)

Role-based approval routing (Manager vs Finance RBAC)

Cost + token usage monitoring for Azure OpenAI
