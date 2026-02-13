import argparse
import collections
import gzip
import html
import json
import os
import re
import datetime
from tqdm import tqdm

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', str(text))
    text = html.unescape(text)
    text = text.replace("&quot;", "\"").replace("&amp;", "&")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def check_path(path):
    os.makedirs(path, exist_ok=True)

def write_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def write_remap_index(index_map, file_path):
    with open(file_path, 'w') as f:
        for original, mapped in index_map.items():
            f.write(f"{original}\t{mapped}\n")

def get_timestamp_start(year, month):
    return int(datetime.datetime(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0).timestamp())

amazon18_dataset2fullname = {
    'Industrial': 'Industrial_and_Scientific',
    'Office': 'Office_Products',
}


def load_metadata_hybrid(args):
    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    metadata_file = os.path.join(args.input_path, 'Metadata', f'meta_{dataset_full_name}.json.gz')
    
    metadata = []
    id_title = {}
    remove_items = set()

    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} not found")
        return [], {}, set()

    with gzip.open(metadata_file, "r") as fp:
        for line in tqdm(fp, desc="Loading Metas (Hybrid)"):
            try:
                data = json.loads(line)
                
                if 'title' not in data:
                    remove_items.add(data['asin'])
                    continue
                    
                title = clean_text(data["title"])
                data['title'] = title

                if len(title) > 1 and len(title.split(" ")) <= 20:
                    id_title[data['asin']] = title
                    metadata.append(data)
                else:
                    remove_items.add(data['asin'])
            except ValueError:
                continue
                
    return metadata, id_title, remove_items

def load_reviews_hybrid(args):

    dataset_full_name = amazon18_dataset2fullname[args.dataset]

    rating_file_path = os.path.join(args.input_path, 'Ratings', dataset_full_name + '.csv')
    
    reviews = []
    
    if not os.path.exists(rating_file_path):
        print(f"Rating file {rating_file_path} not found")
        return []

    with open(rating_file_path, 'r') as fp:
        for line in tqdm(fp, desc='Load ratings (Hybrid)'):
            try:
                item, user, rating, time = line.strip().split(',')
                
                review_obj = {
                    'reviewerID': user,
                    'asin': item,
                    'overall': float(rating),
                    'unixReviewTime': int(time)
                }
                reviews.append(review_obj)
            except ValueError:
                continue
                
    return reviews

def load_review_text_hybrid(args, user2index, item2index):

    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    review_file_path = os.path.join(args.input_path, 'Review', dataset_full_name + '.json.gz')
    
    review_data = {}
    
    if not os.path.exists(review_file_path):
        print("Review text file not found, skipping review content.")
        return {}

    with gzip.open(review_file_path, "r") as fp:
        for line in tqdm(fp, desc='Load review text'):
            inter = json.loads(line)
            try:
                user = inter['reviewerID']
                item = inter['asin']
                
                if user in user2index and item in item2index:
                    uid = user2index[user]
                    iid = item2index[item]
                    
                    if 'unixReviewTime' in inter:
                        timestamp = inter['unixReviewTime']
                        unique_key = str((uid, iid, timestamp))
                    else:
                        continue
                    
                    review_text = clean_text(inter.get('reviewText', ''))
                    summary = clean_text(inter.get('summary', ''))
                    
                    review_data[unique_key] = {"review": review_text, "summary": summary}

            except ValueError:
                continue

    return review_data


def k_core_filtering_json2csv_style(reviews, id_title, K=5, start_timestamp=None, end_timestamp=None):

    remove_users = set()
    remove_items = set()
    
    for review in reviews:
        if review['asin'] not in id_title:
            remove_items.add(review['asin'])
    
    while True:
        new_reviews = []
        flag = False
        total = 0
        user_counts = dict()
        item_counts = dict()
        
        for review in tqdm(reviews, desc="K-core filtering"):

            if start_timestamp and end_timestamp:
                if int(review["unixReviewTime"]) < start_timestamp or int(review["unixReviewTime"]) > end_timestamp:
                    continue
            
            if review['reviewerID'] in remove_users or review['asin'] in remove_items:
                continue
            
            if review['reviewerID'] not in user_counts:
                user_counts[review['reviewerID']] = 0
            user_counts[review['reviewerID']] += 1
            
            if review['asin'] not in item_counts:
                item_counts[review['asin']] = 0
            item_counts[review['asin']] += 1
            
            total += 1
            new_reviews.append(review)
        
        for user in user_counts:
            if user_counts[user] < K:
                remove_users.add(user)
                flag = True
        
        for item in item_counts:
            if item_counts[item] < K:
                remove_items.add(item)
                flag = True
        
        if len(user_counts) > 0 and len(item_counts) > 0:
            density = total / (len(user_counts) * len(item_counts))
        else:
            density = 0
            
        print(f"Users: {len(user_counts)}, Items: {len(item_counts)}, Reviews: {total}, Density: {density}")
        
        if not flag:
            break
        
        reviews = new_reviews
    
    return new_reviews, user_counts, item_counts

def process_dataset_recursive(args, reviews, start_timestamp, end_timestamp):

    metadata, id_title, _ = load_metadata_hybrid(args)
    
    if not metadata:
        print(f"Error: No metadata found for dataset {args.dataset}")
        return None
    
    print(f"Loaded {len(metadata)} metadata items with valid titles (<= 20 words)")
    
    print("Performing k-core filtering...")
    filtered_reviews, user_counts, item_counts = k_core_filtering_json2csv_style(
        reviews, id_title, args.user_k, start_timestamp, end_timestamp
    )
    
    print(f"After filtering: {len(user_counts)} users, {len(item_counts)} items")
    
    if args.st_year > 1996 and len(item_counts) < 2000:
        print(f"Items count {len(item_counts)} < 2000, expanding time range...")
        args.st_year -= 1
        new_start_timestamp = get_timestamp_start(args.st_year, args.st_month)
        print(f"New time range: {args.st_year}-{args.st_month} to {args.ed_year}-{args.ed_month}")
        return process_dataset_recursive(args, reviews, new_start_timestamp, end_timestamp)
    
    return filtered_reviews, user_counts, item_counts, metadata, id_title

def convert_inters2dict_amazon18_style(reviews):

    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    
    user_reviews = collections.defaultdict(list)
    for review in reviews:
        user_reviews[review['reviewerID']].append(review)
    
    for user in user_reviews:
        user_reviews[user].sort(key=lambda x: int(x['unixReviewTime']))
    
    interactions = []
    for user in user_reviews:
        if user not in user2index:
            user2index[user] = len(user2index)
        
        user_items = []
        for review in user_reviews[user]:
            item = review['asin']
            if item not in item2index:
                item2index[item] = len(item2index)
            
            user_items.append(item)
            interactions.append((user, item, float(review['overall']), int(review['unixReviewTime'])))
        
        user2items[user2index[user]] = [item2index[item] for item in user_items]
    
    return user2items, user2index, item2index, interactions

def generate_interaction_list_json2csv_style(reviews, user2index, item2index, id_title):

    interact = dict()
    item2id = {item: idx for item, idx in item2index.items()}
    
    for review in tqdm(reviews, desc="Building interaction list"):
        user = review['reviewerID']
        item = review['asin']
        
        if user not in interact:
            interact[user] = {'items': [], 'ratings': [], 'timestamps': [], 'item_ids': [], 'titles': []}
        
        interact[user]['items'].append(item)
        interact[user]['ratings'].append(review['overall'])
        interact[user]['timestamps'].append(review['unixReviewTime'])
        interact[user]['item_ids'].append(item2id[item])

        interact[user]['titles'].append(id_title[item])
    
    interaction_list = []
    for user in tqdm(interact.keys(), desc="Creating interaction sequences"):
        items = interact[user]['items']
        ratings = interact[user]['ratings']
        timestamps = interact[user]['timestamps']
        item_ids = interact[user]['item_ids']
        titles = interact[user]['titles']
        
        all_data = list(zip(items, ratings, timestamps, item_ids, titles))
        all_data.sort(key=lambda x: int(x[2]))
        items, ratings, timestamps, item_ids, titles = zip(*all_data)
        
        for i in range(1, len(items)):
            st = max(i - 10, 0)
            interaction_list.append([
                user, items[st:i], items[i], item_ids[st:i], item_ids[i],
                titles[st:i], titles[i], ratings[st:i], ratings[i],
                timestamps[st:i], timestamps[i]
            ])
    
    interaction_list.sort(key=lambda x: int(x[-1]))
    return interaction_list

def convert_to_atomic_files_json2csv_style(args, interaction_list, user2index):

    print('Convert dataset (Global 8:1:1): ')
    check_path(os.path.join(args.output_path, args.dataset))
    
    total_len = len(interaction_list)
    train_end = int(total_len * 0.8)
    valid_end = int(total_len * 0.9)
    
    train_interactions = interaction_list[:train_end]
    valid_interactions = interaction_list[train_end:valid_end]
    test_interactions = interaction_list[valid_end:]
    
    def write_file(filename, data):
        with open(os.path.join(args.output_path, args.dataset, filename), 'w') as file:
            file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
            for interaction in data:
                user_id = user2index[interaction[0]]
                history_item_ids = [str(x) for x in interaction[3]]
                target_item_id = str(interaction[4])
                # Limit output history to 50
                history_seq = history_item_ids[-50:]
                file.write(f'{user_id}\t{" ".join(history_seq)}\t{target_item_id}\n')

    write_file(f'{args.dataset}.train.inter', train_interactions)
    write_file(f'{args.dataset}.valid.inter', valid_interactions)
    write_file(f'{args.dataset}.test.inter', test_interactions)
    
    return train_interactions, valid_interactions, test_interactions

def create_item_features_amazon18_style(metadata, item2index, id_title):

    item2feature = collections.defaultdict(dict)
    asin_to_meta = {meta['asin']: meta for meta in metadata}
    
    for item_asin, item_id in item2index.items():
        if item_asin in asin_to_meta:
            meta = asin_to_meta[item_asin]
            title = id_title.get(item_asin, clean_text(meta.get("title", "")))
            
            categories = meta.get("category", [])
            new_categories = []
            if isinstance(categories, list):
                for cat in categories:
                    if isinstance(cat, str) and "</span>" not in cat:
                        new_categories.append(cat.strip())
            categories_str = ",".join(new_categories).strip()
            
            item2feature[item_id] = {
                "title": title,
                "description": clean_text(meta.get("description", "")),
                "brand": meta.get("brand", "").replace("by\n", "").strip(),
                "categories": categories_str
            }
    return item2feature


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, required=True, help='input path containing Ratings/, Metadata/')
    parser.add_argument('--dataset', type=str, default='Arts', help='dataset name key')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--st_year', type=int, default=1996, help='start year')
    parser.add_argument('--st_month', type=int, default=10, help='start month')
    parser.add_argument('--ed_year', type=int, default=2018, help='end year')
    parser.add_argument('--ed_month', type=int, default=11, help='end month')
    parser.add_argument('--output_path', type=str, default='./data', help='output directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    print(f'Processing dataset: {args.dataset}')
    print(f'Input Path: {args.input_path}')
    
    start_timestamp = get_timestamp_start(args.st_year, args.st_month)
    end_timestamp = get_timestamp_start(args.ed_year, args.ed_month)
    
    print("Loading ratings from CSV...")
    reviews = load_reviews_hybrid(args)
    if not reviews:
        exit(1)
    print(f"Loaded {len(reviews)} raw ratings.")
    
    result = process_dataset_recursive(args, reviews, start_timestamp, end_timestamp)
    if result is None:
        exit(1)
    filtered_reviews, user_counts, item_counts, metadata, id_title = result
    
    print("Converting to dict and indexing...")
    user2items, user2index, item2index, interactions = convert_inters2dict_amazon18_style(filtered_reviews)
    
    print("Generating global interaction list (8:1:1)...")
    interaction_list = generate_interaction_list_json2csv_style(filtered_reviews, user2index, item2index, id_title)
    
    train, valid, test = convert_to_atomic_files_json2csv_style(args, interaction_list, user2index)
    
    user2items_final = {u: l for u, l in user2items.items()}
    write_json_file(user2items_final, os.path.join(args.output_path, args.dataset, f'{args.dataset}.inter.json'))
    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2id'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2id'))

    print("Creating item features...")
    item2feature = create_item_features_amazon18_style(metadata, item2index, id_title)
    write_json_file(item2feature, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item.json'))
    
    print("Loading review text from Review/ JSON.gz...")
    review_data = load_review_text_hybrid(args, user2index, item2index)
    write_json_file(review_data, os.path.join(args.output_path, args.dataset, f'{args.dataset}.review.json'))
    
    print("Processing completed!")