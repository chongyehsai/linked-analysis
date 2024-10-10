import streamlit as st
import pandas as pd
import json
import ast  # Import ast for literal evaluation
from collections import Counter

# Configure the page
st.set_page_config(page_title='Game Analysis App - Home', page_icon=":house:")
st.title("Game Data Analysis Tool :game_die:")

# File uploader for CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
page = st.sidebar.selectbox("Select Filter Page", ["Individual Filter", "Related Group Filter"])

# Function to safely evaluate and keep list-like columns intact
def evaluate_list_column(df, column):
    def safe_eval(value):
        try:
            if isinstance(value, str):
                return ast.literal_eval(value)
            return value
        except (ValueError, SyntaxError):
            return value
    
    df[column] = df[column].apply(safe_eval)
    return df

# Format group keys for better readability
def format_group_keys(columns, keys):
    formatted_keys = []
    for col, key in zip(columns, keys):
        if isinstance(key, tuple):
            formatted_keys.append(f"{col}: {', '.join(key)}")
        else:
            formatted_keys.append(f"{col}: {key}")
    return "; ".join(formatted_keys)

if uploaded_file is not None:
    try:
        # Load CSV in chunks and immediately filter usernames to save memory
        chunk_size = 10000
        df = pd.concat(
            [chunk.assign(username=lambda x: x['username'].str.lstrip('0'))
             for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size, dtype={'username': str})],
            ignore_index=True
        )
        
        st.write(f"Sample Data (Total rows: {df.shape[0]}):")
        st.dataframe(df.head(10))

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Apply evaluate_list_column to specified list-like columns without exploding them
    list_columns = ['ip', 'registered_ip', 'hash_password', 'device_id', 'rng']
    for col in list_columns:
        if col in df.columns:
            df = evaluate_list_column(df, col)
            df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    # Individual Filter Page
    if page == "Individual Filter":
        st.subheader("Individual Filter")
        
        # Sidebar filters for individual metrics
        unique_number_count = st.sidebar.slider("Unique Number Count", 0, 100, (70, 100))
        min_average_cost = st.sidebar.slider("Minimum Average Cost", 0, 100, 0)
        user_profit_rate = st.sidebar.slider("User Profit Rate (%)", 0, 100, (0, 10))
        min_user_win_lose = st.sidebar.number_input("Minimum User Win/Lose", -1000, 1000, 0)
        max_bet_amount_range = st.sidebar.slider("Maximum Cost Difference", 0, 100, 50)

        apply_filter = st.sidebar.button("Apply Filter")
        
        required_columns = {'average_cost', 'unique_number_count', 'user_profit_rate', 'user_win_lose', 'number_cost'}
        if required_columns.issubset(df.columns):
            if apply_filter:
                # Using ast.literal_eval for bet amount range calculation to handle non-JSON conforming strings
                df['bet_amount_range'] = df['number_cost'].apply(
                    lambda nc: max(ast.literal_eval(nc).values()) - min(ast.literal_eval(nc).values()) if nc else 0
                )

                # Filter data based on input criteria
                filtered_df = df[
                    df['unique_number_count'].between(*unique_number_count) &
                    (df['average_cost'] >= min_average_cost) &
                    df['user_profit_rate'].between(*user_profit_rate) &
                    (df['user_win_lose'] >= min_user_win_lose) &
                    (df['bet_amount_range'] <= max_bet_amount_range)
                ]

                st.write(f"Filtered Data (Total rows: {filtered_df.shape[0]}):")
                st.dataframe(filtered_df)  # Display the dataframe with list values intact

                # Download option for filtered data
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Filtered Data as CSV", csv, 'filtered_individual.csv', 'text/csv')
        else:
            st.warning("Missing required columns. Please check your dataset.")

    # Related Group Filter Page
    elif page == "Related Group Filter":
        st.subheader("Related Group Filter")

        # Sidebar selection for filter criteria with only IP selected by default
        filter_criteria = {
            'IP': 'ip', 
            'Registered IP': 'registered_ip', 
            'Hash Password': 'hash_password', 
            'Device ID': 'device_id', 
            'RNG': 'rng'
        }
        selected_columns = [col for label, col in filter_criteria.items() if st.sidebar.checkbox(f"Filter by {label}", value=(label == 'IP'))]

        # Pre-grouping filters for individual data
        st.sidebar.markdown("### Pre-Grouping Filters")
        pre_group_min_avg_cost = st.sidebar.slider("Minimum Individual Average Cost before Grouping", 0, 100, 0)
        pre_group_min_unique_count = st.sidebar.slider("Minimum Individual Unique Number Count before Grouping", 0, 100, 10)

        # Sidebar filters for aggregated group data
        st.sidebar.markdown("### Group Filter Criteria")
        group_unique_number_count = st.sidebar.slider("Group Unique Number Count", 0, 100, (70, 100))
        group_min_average_cost = st.sidebar.slider("Group Minimum Average Cost", 0, 100, 0)
        group_user_profit_rate = st.sidebar.slider("Group Profit Rate (%)", 0, 100, (0, 10))
        group_min_user_win_lose = st.sidebar.number_input("Minimum Group Win/Lose", -10000, 10000, 0)
        max_cost_difference = st.sidebar.slider("Maximum Cost Difference", 0, 100, 50)

        apply_filter = st.sidebar.button("Apply Filter")

        if apply_filter and selected_columns:
            try:
                # Apply pre-grouping filters
                pre_filtered_df = df[
                    (df['average_cost'] > pre_group_min_avg_cost) &
                    (df['unique_number_count'] > pre_group_min_unique_count)
                ]

                # Group by selected columns and aggregate data
                grouped = pre_filtered_df.groupby(selected_columns).agg(
                    username_count=('username', 'nunique'),
                    total_cost=('total_cost', 'sum'),
                    total_rewards=('rewards', 'sum'),
                    combined_number_cost=('number_cost', lambda x: Counter(sum((ast.literal_eval(nc) for nc in x), Counter())))
                )

                # Calculate metrics for groups
                grouped['unique_number_count'] = grouped['combined_number_cost'].apply(len)
                grouped['average_cost'] = grouped['total_cost'] / grouped['unique_number_count']
                grouped['user_win_lose'] = grouped['total_rewards'] - grouped['total_cost']
                grouped['user_profit_rate'] = (grouped['user_win_lose'] / grouped['total_rewards'] * 100).fillna(0)
                grouped['cost_range'] = grouped['combined_number_cost'].apply(lambda nc: max(nc.values()) - min(nc.values()) if nc else 0)

                # Filter aggregated group data
                filtered_grouped = grouped[
                    (group_unique_number_count[0] <= grouped['unique_number_count']) &
                    (grouped['unique_number_count'] <= group_unique_number_count[1]) &
                    (grouped['average_cost'] >= group_min_average_cost) &
                    (group_user_profit_rate[0] <= grouped['user_profit_rate']) &
                    (grouped['user_profit_rate'] <= group_user_profit_rate[1]) &
                    (grouped['user_win_lose'] >= group_min_user_win_lose) &
                    (grouped['cost_range'] <= max_cost_difference)
                ]

                if not filtered_grouped.empty:
                    st.write("Filtered Combined Group Information:")
                    st.dataframe(filtered_grouped)

                    # Download combined group data as CSV
                    csv_combined = filtered_grouped.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Combined Group Data as CSV", csv_combined, 'filtered_combined_group_related.csv', 'text/csv')
                else:
                    st.info("No groups met the specified criteria.")
                    
            except KeyError as e:
                st.warning(f"Missing column: {e}")

else:
    st.write("Please upload a CSV file.")
