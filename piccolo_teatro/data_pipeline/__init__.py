from .ingestion import Sales, Products, Seasons
from .transformation import add_features

def clean_data(SALES, PRODUCTS, SEASONS):
    sales = Sales(SALES)
    products = Products(PRODUCTS)
    seasons = Seasons(SEASONS)

    sales.clean()
    products.clean()
    seasons.clean()
    return sales, products, seasons

def run_data_pipeline(SALES, PRODUCTS, SEASONS, show_id=None):
    """Ingesting and cleaning data"""
    
    sales, products, seasons = clean_data(SALES, PRODUCTS, SEASONS)
    if show_id is not None:
        sales.get_same_show(show_id)

    df = add_features(seasons=seasons,
                    products=products,
                    df=sales.df,
                    fill_to_show_date=False)
    
    return df