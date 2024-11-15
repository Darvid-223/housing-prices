import pandas as pd
from src.data_management import get_cleaned_data

# Ladda en dataset (använd "default" eller relevant källa)
X_live_sale_price = get_cleaned_data("default")

# Kontrollera att kolumnerna finns i data
print(X_live_sale_price[['1stFlrSF', 'LotArea', 'GrLivArea']].describe())
