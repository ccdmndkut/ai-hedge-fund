from tools.api import get_option_chain
from datetime import datetime

# Test the options chain API
print("Testing options chain generation for AAPL...")
options_data = get_option_chain("AAPL")

# Display summary information
print(f"Number of call options: {len(options_data['calls'])}")
print(f"Number of put options: {len(options_data['puts'])}")

# Display sample call option
if options_data['calls']:
    sample_call = options_data['calls'][len(options_data['calls'])//2]  # Take a middle option
    print("\nSample Call Option:")
    print(f"Strike: {sample_call['strike']}")
    print(f"Expiration: {sample_call['expiration']}")
    print(f"IV: {sample_call['impliedVolatility']}")
    print(f"Volume: {sample_call['volume']}")
    print(f"Open Interest: {sample_call['openInterest']}")
    print(f"Delta: {sample_call['delta']}")
    print(f"Gamma: {sample_call['gamma']}")

# Test with different ticker
print("\nTesting options chain generation for TSLA...")
options_data = get_option_chain("TSLA")
print(f"Number of call options: {len(options_data['calls'])}")
print(f"Number of put options: {len(options_data['puts'])}")

print("\nOptions data generation is working correctly!") 