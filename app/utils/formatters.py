from typing import Union

def format_inr(amount: Union[float, int]) -> str:
    """
    Formats a numeric amount into Indian digit grouping with 2 decimal places:
    e.g. 849465.44 -> ₹8,49,465.44
         12651781.04 -> ₹1,26,51,781.04
         500.00 -> ₹500.00
    """
    amt_val = float(amount)
    is_negative = amt_val < 0
    abs_amt = abs(amt_val)
    formatted_num = f"{abs_amt:.2f}"
    integer_part, decimal_part = formatted_num.split(".")

    if len(integer_part) <= 3:
        grouped = integer_part
    else:
        last3 = integer_part[-3:]
        remaining = integer_part[:-3]
        parts = []
        while len(remaining) > 2:
            parts.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            parts.insert(0, remaining)
        parts.append(last3)
        grouped = ",".join(parts)

    prefix = "-₹" if is_negative else "₹"
    return f"{prefix}{grouped}.{decimal_part}"

def format_inr_lakh_crore(amount: Union[float, int]) -> str:
    """
    Returns concise human-friendly Indian currency terms:
    e.g. 125000 -> ₹1.25 lakh
         12651781.04 -> ₹1.27 crore
         45000 -> ₹45,000
    """
    val = float(amount)
    abs_val = abs(val)
    prefix = "-" if val < 0 else ""

    if abs_val >= 10_000_000:  # 1 Crore = 100 Lakhs
        crores = abs_val / 10_000_000
        return f"{prefix}₹{crores:.2f} crore"
    elif abs_val >= 100_000:   # 1 Lakh = 100,000
        lakhs = abs_val / 100_000
        return f"{prefix}₹{lakhs:.2f} lakh"
    else:
        return format_inr(amount)
