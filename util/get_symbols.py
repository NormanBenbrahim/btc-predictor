from binance.client import Client
import os 

KEY = os.environ['BINANCE_API_KEY']
SECRET_KEY = os.environ['BINANCE_SECRET_KEY']

client = Client(KEY, SECRET_KEY)

# list of all available symbols
full_list = client.get_symbol_ticker()

with open('./data/binance-symbols.txt', 'w+') as f:
    for symbol in full_list:
        f.writelines(symbol['symbol'] + '\n')
        print(symbol['symbol'])