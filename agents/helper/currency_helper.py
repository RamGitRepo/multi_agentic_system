import os
import dotenv
dotenv.load_dotenv()


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_ENDPOINT = "https://api.tavily.com/search"

def _extract_price_currency(text: str, currency_hint: str = "£"):
    """
    Heuristic price extractor. Returns (amount: float, currency: str) or (None, None).
    Looks for £, $, €, or currency codes GBP/EUR/USD.
    """
    if not text:
        return None, None
    # Common patterns, favor currency_hint occurrences if present
    patterns = [
        r"(?i)\b(GBP|EUR|USD)\s*([0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]{2})?)",
        r"(£|\$|€)\s*([0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]{2})?)"
    ]

    import re
    def _normalize_amt(s: str) -> float:
        return float(s.replace(",", "").replace(" ", ""))

    best = None
    for pat in patterns:
        for m in re.finditer(pat, text):
            cur = m.group(1)
            amt = m.group(2)
            if cur in ("£", "$", "€"):
                cur = {"£": "GBP", "$": "USD", "€": "EUR"}[cur]
            try:
                val = _normalize_amt(amt)
            except Exception:
                continue
            cand = (val, cur)
            # prefer currency_hint
            if cur.upper() == (currency_hint or "").upper():
                return cand
            if best is None:
                best = cand
    return best if best else (None, None)