from pydantic import BaseModel

class DfColumns(BaseModel):
    const: bool
    enabled: bool

a = DfColumns(const=2, enabled=True)