from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict, Optional

class Decision(Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"

class TransactionRequest(BaseModel):
    creation_date: str = Field(..., alias="Creation date")
    country: str = Field(..., alias="Country")
    valor: float = Field(..., alias="Valor")
    processing_value: float = Field(..., alias="Processing value")
    merchant_id: str = Field(..., alias="Merchant Id")
    account_id: str = Field(..., alias="Account Id")
    bin_bank: str = Field(..., alias="BIN Bank")
    country_bin_iso: str = Field(..., alias="Country BIN ISO")
    card_type: str = Field(..., alias="Card type")
    franchise: str = Field(..., alias="Franchise")
    transaction_type: str = Field(..., alias="Transaction type")
    payment_method: str = Field(..., alias="Payment method")
    payment_model: str = Field(..., alias="Payment model")
    transaction_origin: str = Field(..., alias="Transaction origin")
    accreditation_model: str = Field(..., alias="Accreditation model")
    transaction_currency: str = Field(..., alias="Transaction currency")
    processing_currency: str = Field(..., alias="Processing currency")
    visible_number: str = Field(..., alias="Visible number")

    class Config:
        populate_by_name = True

class FraudResponse(BaseModel):
    transaction_id: str
    score: float
    decision: Decision
    thresholds: Dict[str, float]
    latency_ms: float = 0.0