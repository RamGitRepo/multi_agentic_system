# --- Azure Blob upload (no SAS) ---
import os
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContentSettings
PUBLIC_API_BASE = os.getenv("PUBLIC_API_BASE")
load_dotenv()

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdf_canvas
except Exception:
    A4_pagesize = None
    pdf_canvas = None


# Azure Blob config from env vars
AZURE_BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
AZURE_BLOB_ACCOUNT_NAME = os.getenv("AZURE_BLOB_ACCOUNT_NAME")
AZURE_BLOB_ACCOUNT_KEY = os.getenv("AZURE_BLOB_ACCOUNT_KEY")
PO_BLOB_CONTAINER = os.getenv("BLOB_CONTAINER")



def _get_blob_service_client() -> BlobServiceClient:
    if AZURE_BLOB_CONNECTION_STRING:
        return BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
    if AZURE_BLOB_ACCOUNT_NAME and AZURE_BLOB_ACCOUNT_KEY:
        return BlobServiceClient(
            account_url=f"https://{AZURE_BLOB_ACCOUNT_NAME}.blob.core.windows.net",
            credential=AZURE_BLOB_ACCOUNT_KEY,
        )
    raise RuntimeError(
        "Azure Blob credentials not found. "
        "Set AZURE_BLOB_CONNECTION_STRING or AZURE_BLOB_ACCOUNT_NAME + AZURE_BLOB_ACCOUNT_KEY."
    )

def _make_po_doc_bytes(
     po_id: str, Product: str, quantity: int, unit_price: float, total_price: float,
    vendor_name: str, vendor_url: str
) -> tuple[bytes, str, str]:
    """Build a PDF (ReportLab) or TXT fallback entirely in memory."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

    if pdf_canvas is not None and A4 is not None:
        buf = BytesIO()
        c = pdf_canvas.Canvas(buf, pagesize=A4)
        w, h = A4; y = h - 50
        def line(txt: str): nonlocal y; c.drawString(50, y, txt); y -= 18
        c.setFont("Helvetica-Bold", 16); c.drawString(50, y, "PURCHASE ORDER"); y -= 28
        c.setFont("Helvetica", 11)
        for txt in [
            f"PO ID: {po_id}",
            f"Date: {ts}",
            f"Item: {Product}",
            f"Quantity: {quantity}",
            f"Unit Price: £{unit_price:.2f}",
            f"Total Price: £{total_price:.2f}",
            f"Vendor Name: {vendor_name}",
            f"Vendor URL: {vendor_url}",
            "Status: PLACED",
        ]: line(txt)
        c.showPage(); c.save()
        return buf.getvalue(), f"{po_id}.pdf", "application/pdf"

    # TXT fallback
    txt = (
        "PURCHASE ORDER\n\n"
        f"PO ID: {po_id}\n"
        f"Date: {ts}\n"
        f"Item: {Product}\n"
        f"Quantity: {quantity}\n"
        f"Unit Price: £{unit_price:.2f}\n"
        f"Total Price: £{total_price:.2f}\n"
        f"Vendor Name: {vendor_name}\n"
        f"Vendor URL: {vendor_url}\n"
        "Status: PLACED\n"
    )
    return txt.encode("utf-8"), f"{po_id}.txt", "text/plain"

def generate_po_pdf(
    po_id: str, Product: str, quantity: int, unit_price: float, total_price: float,
    vendor_name: str, vendor_url: str
) -> dict:
    """
    Generate the PO document, upload to Azure Blob (private container),
    and return the API URL your UI can open: /po/{po_id}/pdf
    """
    blob_bytes, filename, content_type = _make_po_doc_bytes(
        po_id=po_id, Product=Product, quantity=quantity, unit_price=unit_price,
        total_price=total_price, vendor_name=vendor_name, vendor_url=vendor_url
    )

    svc = _get_blob_service_client()
    container = svc.get_container_client(PO_BLOB_CONTAINER) # type: ignore
    try:
        container.create_container()
    except Exception:
        pass  # already exists

    # one doc per PO; if you want versions, add a timestamp suffix
    blob_name = f"{po_id}/{filename}"
    blob_client = container.get_blob_client(blob_name)
    try:
        blob_client.upload_blob(
        data=blob_bytes,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
            metadata={
            "PoNumber": po_id,
            "Product": (Product[:200] if Product else ""),
            "VendorName": (vendor_name[:200] if vendor_name else "N/A"),
        },
        )
    except Exception as ex:
        print(f"❌ Failed to upload PO document to Blob: {ex}")
        raise

    return {
        "api_url": f"/po/{po_id}/pdf",
        "blob_url": blob_client.url,
    }

def _abs_pdf_url(pdf_api_path: str) -> str:
    if pdf_api_path.startswith("http://") or pdf_api_path.startswith("https://"):
        return pdf_api_path
    return f"{PUBLIC_API_BASE}{pdf_api_path}"
