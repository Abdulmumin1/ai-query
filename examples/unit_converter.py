"""Unit Converter Agent

Demonstrates:
- Simple synchronous tool execution
- Mathematical conversions
- Clean tool definitions
"""

import asyncio
from ai_query import generate_text, google, tool, step_count_is


# --- Conversion Tools ---

def convert_temperature(value: float, from_unit: str, to_unit: str):
    """Convert temperature between Celsius, Fahrenheit, and Kelvin."""
    print(f"  [Convert] {value} {from_unit} -> {to_unit}")

    # Normalize units
    from_unit = from_unit.lower()[0]  # c, f, or k
    to_unit = to_unit.lower()[0]

    # Convert to Celsius first
    if from_unit == "f":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "k":
        celsius = value - 273.15
    else:
        celsius = value

    # Convert from Celsius to target
    if to_unit == "f":
        result = celsius * 9 / 5 + 32
        unit_name = "Fahrenheit"
    elif to_unit == "k":
        result = celsius + 273.15
        unit_name = "Kelvin"
    else:
        result = celsius
        unit_name = "Celsius"

    return f"{value} {from_unit.upper()} = {result:.2f} {unit_name}"


def convert_length(value: float, from_unit: str, to_unit: str):
    """Convert length between various units."""
    print(f"  [Convert] {value} {from_unit} -> {to_unit}")

    # Conversion factors to meters
    to_meters = {
        "mm": 0.001, "cm": 0.01, "m": 1, "km": 1000,
        "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.34,
        "inch": 0.0254, "inches": 0.0254, "foot": 0.3048, "feet": 0.3048,
        "yard": 0.9144, "yards": 0.9144, "mile": 1609.34, "miles": 1609.34,
        "meter": 1, "meters": 1, "kilometer": 1000, "kilometers": 1000,
    }

    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    if from_unit not in to_meters or to_unit not in to_meters:
        return f"Unknown unit. Supported: {', '.join(set(to_meters.keys()))}"

    # Convert to meters, then to target
    meters = value * to_meters[from_unit]
    result = meters / to_meters[to_unit]

    return f"{value} {from_unit} = {result:.4f} {to_unit}"


def convert_weight(value: float, from_unit: str, to_unit: str):
    """Convert weight/mass between various units."""
    print(f"  [Convert] {value} {from_unit} -> {to_unit}")

    # Conversion factors to grams
    to_grams = {
        "mg": 0.001, "g": 1, "kg": 1000,
        "oz": 28.3495, "lb": 453.592, "lbs": 453.592,
        "ounce": 28.3495, "ounces": 28.3495,
        "pound": 453.592, "pounds": 453.592,
        "gram": 1, "grams": 1, "kilogram": 1000, "kilograms": 1000,
    }

    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    if from_unit not in to_grams or to_unit not in to_grams:
        return f"Unknown unit. Supported: {', '.join(set(to_grams.keys()))}"

    grams = value * to_grams[from_unit]
    result = grams / to_grams[to_unit]

    return f"{value} {from_unit} = {result:.4f} {to_unit}"


def convert_currency(amount: float, from_currency: str, to_currency: str):
    """Convert currency using approximate rates (for demo purposes)."""
    print(f"  [Convert] {amount} {from_currency} -> {to_currency}")

    # Approximate rates to USD (demo only - use real API for production)
    to_usd = {
        "USD": 1, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067,
        "CAD": 0.74, "AUD": 0.65, "CHF": 1.13, "CNY": 0.14,
        "INR": 0.012, "MXN": 0.058, "BRL": 0.20, "KRW": 0.00075,
    }

    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    if from_currency not in to_usd or to_currency not in to_usd:
        return f"Unknown currency. Supported: {', '.join(to_usd.keys())}"

    usd = amount * to_usd[from_currency]
    result = usd / to_usd[to_currency]

    return f"{amount} {from_currency} ≈ {result:.2f} {to_currency} (approximate rate)"


temp_tool = tool(
    description="Convert temperature between Celsius, Fahrenheit, and Kelvin.",
    parameters={
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "The temperature value"},
            "from_unit": {"type": "string", "description": "Source unit (celsius/fahrenheit/kelvin)"},
            "to_unit": {"type": "string", "description": "Target unit (celsius/fahrenheit/kelvin)"}
        },
        "required": ["value", "from_unit", "to_unit"]
    },
    execute=convert_temperature
)

length_tool = tool(
    description="Convert length between mm, cm, m, km, inches, feet, yards, miles.",
    parameters={
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "The length value"},
            "from_unit": {"type": "string", "description": "Source unit"},
            "to_unit": {"type": "string", "description": "Target unit"}
        },
        "required": ["value", "from_unit", "to_unit"]
    },
    execute=convert_length
)

weight_tool = tool(
    description="Convert weight between mg, g, kg, ounces, pounds.",
    parameters={
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "The weight value"},
            "from_unit": {"type": "string", "description": "Source unit"},
            "to_unit": {"type": "string", "description": "Target unit"}
        },
        "required": ["value", "from_unit", "to_unit"]
    },
    execute=convert_weight
)

currency_tool = tool(
    description="Convert currency between major world currencies (USD, EUR, GBP, JPY, etc.). Uses approximate rates.",
    parameters={
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "The amount to convert"},
            "from_currency": {"type": "string", "description": "Source currency code (e.g., USD)"},
            "to_currency": {"type": "string", "description": "Target currency code (e.g., EUR)"}
        },
        "required": ["amount", "from_currency", "to_currency"]
    },
    execute=convert_currency
)


# --- Main ---

async def main():
    print("Unit Converter Agent")
    print("=" * 50)

    questions = [
        "What is 100 degrees Fahrenheit in Celsius?",
        "Convert 5 miles to kilometers",
        "How many pounds is 75 kilograms?",
        "I have 1000 USD, how much is that in EUR, GBP, and JPY?",
        "If water boils at 100°C and freezes at 0°C, what are those in Fahrenheit?",
    ]

    tools = {
        "convert_temperature": temp_tool,
        "convert_length": length_tool,
        "convert_weight": weight_tool,
        "convert_currency": currency_tool,
    }

    for q in questions:
        print(f"\nQ: {q}")

        result = await generate_text(
            model=google("gemini-2.0-flash"),
            system="You are a helpful unit converter. Use the conversion tools to answer questions accurately.",
            prompt=q,
            tools=tools,
            stop_when=step_count_is(4),
        )

        print(f"A: {result.text}")
        print(f"   [{result.usage.total_tokens} tokens]")


if __name__ == "__main__":
    asyncio.run(main())
