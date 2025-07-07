import pandas as pd
from surprise import Dataset, Reader, SVD

def preprocess(df):
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')
    df['product_id'] = pd.to_numeric(df['product_id'], errors='coerce')
    df['category_id'] = pd.to_numeric(df['category_id'], errors='coerce')

    is_cart = df['event_type'] == 'cart'
    cart_df = df[is_cart].drop_duplicates(subset=['product_id', 'user_id', 'user_session'])
    df = df[~is_cart].copy()
    df = pd.concat([df, cart_df], ignore_index=True)

    if 'category_code' in df.columns:
        parts = df['category_code'].astype(str).str.split('.', expand=True)
        df['category'] = parts[0]
        df['sub_category'] = parts[1] if parts.shape[1] > 1 else None
        df['sub_sub_category'] = parts[2] if parts.shape[1] > 2 else None

    return df

def product_features(df):
    view = df[df['event_type'] == 'view'].groupby('product_id').size()
    cart = df[df['event_type'] == 'cart'].groupby('product_id').size()
    purchase = df[df['event_type'] == 'purchase'].groupby('product_id').size()

    stats = pd.DataFrame({
        'views': view,
        'carts': cart,
        'purchases': purchase
    }).fillna(0)

    stats['purchase_per_view'] = stats['purchases'] / stats['views'].replace(0, 1)
    stats['cart_per_view'] = stats['carts'] / stats['views'].replace(0, 1)
    stats['purchase_per_cart'] = stats['purchases'] / stats['carts'].replace(0, 1)

    category_info = df.groupby('product_id').agg({
        'category_id': 'first',
        'category_code': 'first'
    })

    return stats.join(category_info)

def user_features(df):
    return df.groupby('user_id')['event_type'].value_counts().unstack().fillna(0)

def calculate_interaction_matrix(df):
    interactions = df[df['event_type'] == 'purchase']
    grouped = interactions.groupby(['user_id', 'product_id']).size().reset_index(name='rating')
    return grouped

def simple_als(interactions):
    reader = Reader(rating_scale=(interactions['rating'].min(), interactions['rating'].max()))
    data = Dataset.load_from_df(interactions[['user_id', 'product_id', 'rating']], reader)
    trainset = data.build_full_trainset()

    model = SVD()
    model.fit(trainset)

    return model