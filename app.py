import streamlit as st
import pandas as pd
import ast
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

# Function to parse number_cost safely
def parse_number_cost(nc):
    try:
        if isinstance(nc, dict):
            return nc
        elif isinstance(nc, str):
            return ast.literal_eval(nc)
    except (ValueError, SyntaxError):
        return {}

# Format group keys for better readability
def format_group_keys(columns, keys):
    return "; ".join([f"{col}: {key}" if not isinstance(key, tuple) else f"{col}: {', '.join(key)}" for col, key in zip(columns, keys)])

if uploaded_file:
    try:
        # Load CSV in chunks to avoid memory overload
        chunk_size = 10000
        filtered_chunks = []

        for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size, dtype={'username': str}):
            chunk['username'] = chunk['username'].str.lstrip('0')
            filtered_chunks.append(chunk)

        df = pd.concat(filtered_chunks, ignore_index=True)

        # Display a sample of the data
        st.write(f"Sample Data (Total rows: {df.shape[0]}):")
        st.dataframe(df.head(10))

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Check for missing columns and add them if necessary
    required_columns = {'average_cost', 'unique_number_count', 'user_profit_rate', 'user_win_lose', 'number_cost'}
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0

    # Parse `number_cost` safely and calculate bet amount range
    df['parsed_number_cost'] = df['number_cost'].apply(parse_number_cost)
    df['bet_amount_range'] = df['parsed_number_cost'].apply(
        lambda nc: max(nc.values()) - min(nc.values()) if nc else 0
    )

    # Individual Filter Page
    if page == "Individual Filter":
        st.subheader("Individual Filter")
        
        unique_number_count = st.sidebar.slider("Unique Number Count", 0, 100, (70, 100))
        min_average_cost = st.sidebar.slider("Minimum Average Cost", 0, 100, 0)
        user_profit_rate = st.sidebar.slider("User Profit Rate (%)", 0, 100, (0, 10))
        min_user_win_lose = st.sidebar.number_input("Minimum User Win/Lose", -1000, 1000, 0)
        max_bet_amount_range = st.sidebar.slider("Maximum Cost Difference", 0, 100, 50)

        apply_filter = st.sidebar.button("Apply Filter")
        
        if apply_filter:
            filtered_df = df.query(
                "unique_number_count >= @unique_number_count[0] and unique_number_count <= @unique_number_count[1] and "
                "average_cost >= @min_average_cost and "
                "user_profit_rate >= @user_profit_rate[0] and user_profit_rate <= @user_profit_rate[1] and "
                "user_win_lose >= @min_user_win_lose and "
                "bet_amount_range <= @max_bet_amount_range"
            )

            st.write(f"Filtered Data (Total rows: {filtered_df.shape[0]}):")
            st.dataframe(filtered_df)

            # Download button for filtered data
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Filtered Data as CSV", csv, 'filtered_individual.csv', 'text/csv')

    # Related Group Filter Page
    elif page == "Related Group Filter":
        st.subheader("Related Group Filter")

        filter_criteria = {'IP': 'ip', 'Registered IP': 'registered_ip', 'Hash Password': 'hash_password', 'Device ID': 'device_id', 'RNG': 'rng'}
        selected_columns = [col for label, col in filter_criteria.items() if st.sidebar.checkbox(f"Filter by {label}", value=(label == 'IP'))]

        pre_group_min_avg_cost = st.sidebar.slider("Minimum Individual Average Cost before Grouping", 0, 100, 0)
        pre_group_min_unique_count = st.sidebar.slider("Minimum Individual Unique Number Count before Grouping", 0, 100, 10)

        group_unique_number_count = st.sidebar.slider("Group Unique Number Count", 0, 100, (70, 100))
        group_min_average_cost = st.sidebar.slider("Group Minimum Average Cost", 0, 100, 0)
        group_user_profit_rate = st.sidebar.slider("Group Profit Rate (%)", 0, 100, (0, 10))
        group_min_user_win_lose = st.sidebar.number_input("Minimum Group Win/Lose", -10000, 10000, 0)
        max_cost_difference = st.sidebar.slider("Maximum Cost Difference", 0, 100, 50)

        apply_filter = st.sidebar.button("Apply Filter")

        if apply_filter and selected_columns:
            try:
                pre_filtered_df = df.query("average_cost > @pre_group_min_avg_cost and unique_number_count > @pre_group_min_unique_count")
                grouped_df = pre_filtered_df.groupby(selected_columns).filter(lambda x: x[['username', 'ref_provider']].drop_duplicates().shape[0] > 1)

                if not grouped_df.empty:
                    combined_results = []
                    member_details_list = []

                    grouped = grouped_df.groupby(selected_columns)
                    
                    for group_keys, group_data in grouped:
                        combined_number_cost = Counter()
                        for d in group_data['parsed_number_cost']:
                            combined_number_cost.update(d)

                        sorted_combined_number_cost = dict(sorted(combined_number_cost.items()))
                        cost_values = list(sorted_combined_number_cost.values())
                        cost_range = max(cost_values) - min(cost_values) if cost_values else 0

                        total_cost = group_data['total_cost'].sum()
                        total_rewards = group_data['rewards'].sum()
                        unique_number_count = len(sorted_combined_number_cost)
                        average_cost = total_cost / unique_number_count if unique_number_count > 0 else 0
                        user_win_lose = total_rewards - total_cost
                        user_profit_rate = (user_win_lose / total_rewards * 100) if total_rewards > 0 else 0

                        if (cost_range <= max_cost_difference and
                            group_unique_number_count[0] <= unique_number_count <= group_unique_number_count[1] and
                            average_cost >= group_min_average_cost and
                            group_user_profit_rate[0] <= user_profit_rate <= group_user_profit_rate[1] and
                            user_win_lose >= group_min_user_win_lose):

                            combined_results.append({
                                **{col: key for col, key in zip(selected_columns, group_keys)},
                                'combined_number_cost': sorted_combined_number_cost,
                                'combined_rewards': total_rewards,
                                'combined_total_cost': total_cost,
                                'combined_unique_number_count': unique_number_count,
                                'combined_average_cost': average_cost,
                                'combined_user_win_lose': user_win_lose,
                                'combined_user_profit_rate': user_profit_rate
                            })

                            group_keys_formatted = format_group_keys(selected_columns, group_keys)
                            st.write(f"Group with {group_keys_formatted}")
                            st.dataframe(group_data)
                            member_details_list.append(group_data)
                            st.write("---")

                    if member_details_list:
                        member_details_df = pd.concat(member_details_list, ignore_index=True)
                        csv_member_details = member_details_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Member Details as CSV", csv_member_details, 'member_details.csv', 'text/csv')

                    if combined_results:
                        combined_df = pd.DataFrame(combined_results)
                        st.write("Filtered Combined Group Information:")
                        st.dataframe(combined_df)
                        csv_combined = combined_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Combined Group Data as CSV", csv_combined, 'filtered_combined_group_related.csv', 'text/csv')
                    else:
                        st.info("No groups met the specified criteria.")
                else:
                    st.info("No related records found for the selected criteria.")
            except KeyError as e:
                st.warning(f"Missing column: {e}")

else:
    st.write("Please upload a CSV file.")
