"""Map nflverse position strings to Knowball PositionCode / position_group."""

from __future__ import annotations

# Knowball catalog groups (docs/architecture + ETL brief §3.4)
POSITION_TO_CODE: dict[str, str] = {
    "QB": "QB",
    "RB": "RB",
    "FB": "FB",
    "WR": "WR",
    "TE": "TE",
    "T": "T",
    "OT": "OT",
    "G": "G",
    "OG": "OG",
    "C": "C",
    "OL": "OT",  # generic OL → tackle code still lands in ol group
    "ED": "ED",
    "EDGE": "EDGE",
    "DE": "DE",
    "DT": "DT",
    "NT": "NT",
    "LB": "LB",
    "ILB": "ILB",
    "OLB": "OLB",
    "CB": "CB",
    "FS": "FS",
    "SS": "SS",
    "S": "S",
    "SAF": "S",
    "DB": "CB",
    "MLB": "ILB",
    "DL": "DT",
    "K": "K",
    "P": "P",
    "KR": "KR",
    "PR": "PR",
}

CODE_TO_GROUP: dict[str, str] = {
    "QB": "qb",
    "RB": "backfield",
    "FB": "backfield",
    "WR": "pass_catcher",
    "TE": "pass_catcher",
    "T": "ol",
    "OT": "ol",
    "G": "ol",
    "OG": "ol",
    "C": "ol",
    "ED": "def_front",
    "EDGE": "def_front",
    "DE": "def_front",
    "DT": "def_front",
    "NT": "def_front",
    "LB": "def_front",
    "ILB": "def_front",
    "OLB": "def_front",
    "CB": "secondary",
    "FS": "secondary",
    "SS": "secondary",
    "S": "secondary",
    "K": "kicker",
    "P": "punter",
    "KR": "returner",
    "PR": "returner",
}


def map_position(raw: str | None) -> tuple[str | None, str | None]:
    """Return (PositionCode, position_group) or (None, None) if unmapped."""
    if raw is None:
        return None, None
    key = str(raw).strip().upper()
    code = POSITION_TO_CODE.get(key)
    if code is None:
        return None, None
    return code, CODE_TO_GROUP[code]
