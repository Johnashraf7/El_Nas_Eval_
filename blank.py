


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.markdown('<h1 style="text-align: center; color: #1E3A8A; font-size: 2.5rem;">LLM Evaluation Dashboard</h1>', unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        # Load the actual CSV file
        df = pd.read_csv('data_final.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Display basic info about the loaded data
        st.sidebar.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Check and rename columns if needed (based on your data structure)
        column_mapping = {
            'case_no': 'case_no',
            'case_number': 'case_no',
            'case': 'case_no',
            'annotator': 'annotator',
            'evaluator': 'annotator',
            'response_type': 'response_type',
            'type': 'response_type',
            'alignment': 'alignment',
            'comprehensiveness': 'comprehensiveness',
            'correctness': 'correctness',
            'safety': 'safety',
            'structure': 'structure'
        }
        
        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Check if we have the expected columns
        expected_score_cols = ['alignment', 'comprehensiveness', 'correctness', 'safety', 'structure']
        available_score_cols = []
        
        # Convert score columns to numeric, coercing errors
        for col in expected_score_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                available_score_cols.append(col)
        
        # Check required columns
        required_cols = ['case_no', 'annotator', 'response_type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.sidebar.warning(f"Missing required columns: {missing_cols}")
            st.sidebar.info(f"Available columns: {', '.join(df.columns.tolist())}")
            # Try to use what we have
            if 'case_no' not in df.columns:
                # Create case numbers if missing
                if len(df) > 0:
                    df['case_no'] = df.index % 12 + 1
        
        # Ensure case_no is numeric
        if 'case_no' in df.columns:
            df['case_no'] = pd.to_numeric(df['case_no'], errors='coerce')
            # Ensure case numbers are 1-12
            df['case_no'] = df['case_no'].apply(lambda x: ((x - 1) % 12) + 1 if pd.notna(x) else 1)
        
        # Clean up response_type and annotator
        if 'response_type' in df.columns:
            df['response_type'] = df['response_type'].astype(str).str.strip()
        if 'annotator' in df.columns:
            df['annotator'] = df['annotator'].astype(str).str.strip()
        
        return df, available_score_cols
        
    except FileNotFoundError:
        st.sidebar.error("❌ File 'data_final.csv' not found.")
        return None, []
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {str(e)}")
        return None, []

# Load the data
data, score_cols = load_data()

if data is None:
    st.error("""
    ## ❌ Data Loading Error
    
    Please ensure:
    1. The file `data_final.csv` is in the same directory as this script
    2. The CSV file has the correct format with columns like:
       - `case_no` (or similar)
       - `annotator` (or similar)  
       - `response_type` (or similar)
       - Score columns: `alignment`, `comprehensiveness`, `correctness`, `safety`, `structure`
    
    **Current working directory files:**
    """)
    
    import os
    files = os.listdir('.')
    st.write(files)
    st.stop()

# Display data info in sidebar
st.sidebar.header("📊 Data Info")
st.sidebar.write(f"**Total Rows:** {len(data):,}")
st.sidebar.write(f"**Unique Cases:** {data['case_no'].nunique()}")
st.sidebar.write(f"**Annotators:** {data['annotator'].nunique()}")
st.sidebar.write(f"**Response Types:** {data['response_type'].nunique()}")
st.sidebar.write(f"**Score Columns:** {len(score_cols)}")

# Show sample of data
if st.sidebar.checkbox("Show Data Sample", False):
    st.sidebar.dataframe(data.head())

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Response Type filter
response_types = sorted(data['response_type'].unique()) if 'response_type' in data.columns else []
selected_responses = st.sidebar.multiselect(
    "Response Types",
    options=response_types,
    default=['Gold', 'Anas', 'LLM'] if all(x in response_types for x in ['Gold', 'Anas', 'LLM']) else response_types
)

# Annotator filter
annotators = sorted(data['annotator'].unique()) if 'annotator' in data.columns else []
selected_annotators = st.sidebar.multiselect(
    "Annotators",
    options=annotators,
    default=annotators
)

# Case number filter (1-12 as per your data)
case_numbers = sorted([int(x) for x in data['case_no'].dropna().unique() if 1 <= x <= 12])
selected_cases = st.sidebar.multiselect(
    "Case Numbers (1-12)",
    options=case_numbers,
    default=case_numbers
)

# Score threshold filter
score_threshold = st.sidebar.slider(
    "Minimum Average Score to Display",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1
)

# Apply filters
filter_mask = pd.Series([True] * len(data), index=data.index)

if selected_responses:
    filter_mask = filter_mask & data['response_type'].isin(selected_responses)

if selected_annotators:
    filter_mask = filter_mask & data['annotator'].isin(selected_annotators)

if selected_cases:
    filter_mask = filter_mask & data['case_no'].isin(selected_cases)

filtered_data = data[filter_mask].copy()

# Calculate average score for each row and apply threshold filter
if len(score_cols) > 0:
    filtered_data['avg_score'] = filtered_data[score_cols].mean(axis=1, skipna=True)
    if score_threshold > 0:
        filtered_data = filtered_data[filtered_data['avg_score'] >= score_threshold]

# Main dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🧮 Score Analysis", "👥 Annotator Agreement", "📋 Case Details", "📊 Heatmap Analysis"])

with tab1:
    st.subheader("📊 Overall Performance Summary")
    
    if len(filtered_data) == 0:
        st.warning("No data available with current filters.")
    else:
        # Calculate key metrics
        total_cases = filtered_data['case_no'].nunique()
        total_evaluations = len(filtered_data)
        valid_scores = filtered_data[score_cols].notna().sum().sum()
        
        # Calculate statistics by response type
        response_counts = filtered_data['response_type'].value_counts()
        
        # Display metrics in a clean layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cases", total_cases)
            st.metric("Total Evaluations", total_evaluations)
        
        with col2:
            st.metric("Valid Scores", f"{valid_scores:,}")
            st.metric("Evaluation Criteria", len(score_cols))
        
        with col3:
            gold_count = response_counts.get('Gold', 0)
            anas_count = response_counts.get('Anas', 0)
            st.metric("Gold Responses", gold_count)
            st.metric("Anas Responses", anas_count)
        
        with col4:
            llm_count = response_counts.get('LLM', 0)
            st.metric("LLM Responses", llm_count)
            st.metric("Annotators", len(selected_annotators))
        
        # Overall score distribution
        st.subheader("Overall Score Distribution")
        
        # Calculate score distribution
        all_scores = filtered_data[score_cols].values.flatten()
        all_scores_clean = all_scores[~np.isnan(all_scores)]
        
        if len(all_scores_clean) > 0:
            score_counts = pd.Series(all_scores_clean).round(1).value_counts().reindex([0.0, 0.5, 1.0], fill_value=0)
            
            # Create score labels
            score_labels = []
            score_values = []
            colors = []
            
            for score, count in score_counts.items():
                if score == 0.0:
                    label = "0.0 (Poor)"
                    color = '#EF4444'
                elif score == 0.5:
                    label = "0.5 (Average)"
                    color = '#F59E0B'
                elif score == 1.0:
                    label = "1.0 (Excellent)"
                    color = '#10B981'
                else:
                    label = f"{score}"
                    color = '#6B7280'
                
                if count > 0:
                    score_labels.append(label)
                    score_values.append(count)
                    colors.append(color)
            
            if score_values:
                fig_overall = px.bar(
                    x=score_labels,
                    y=score_values,
                    color=score_labels,
                    color_discrete_map={label: color for label, color in zip(score_labels, colors)},
                    title=f'Overall Score Distribution ({len(all_scores_clean):,} scores)',
                    labels={'x': 'Score', 'y': 'Count'},
                    text_auto=True
                )
                fig_overall.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_overall, use_container_width=True)
            else:
                st.info("No score data available for visualization.")
        else:
            st.info("No valid score data available.")
        
        # Performance by response type
        st.subheader("Average Scores by Response Type")
        
        if len(score_cols) > 0:
            summary_stats = filtered_data.groupby('response_type')[score_cols].mean()
            
            if not summary_stats.empty:
                avg_scores = summary_stats.mean(axis=1)
                
                # Sort by response type for consistent ordering
                response_order = ['Gold', 'Anas', 'LLM']
                avg_scores = avg_scores.reindex([rt for rt in response_order if rt in avg_scores.index])
                
                fig_response = px.bar(
                    x=avg_scores.index,
                    y=avg_scores.values,
                    color=avg_scores.index,
                    color_discrete_map={'Gold': '#FFD700', 'Anas': '#2E86AB', 'LLM': '#A23B72'},
                    title='Average Scores by Response Type',
                    labels={'x': 'Response Type', 'y': 'Average Score'},
                    text_auto='.3f'
                )
                fig_response.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_response, use_container_width=True)
            else:
                st.info("No data available for response type comparison.")
        else:
            st.info("No score columns available.")

with tab2:
    st.subheader("🧮 Detailed Score Analysis")
    
    if len(filtered_data) == 0:
        st.warning("No data available with current filters.")
    else:
        # Create tabs for different analyses within this tab
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Score Distribution", "Performance by Criterion", "Case Performance"])
        
        with sub_tab1:
            if len(score_cols) > 0:
                # Calculate number of rows and columns for subplots
                n_plots = len(score_cols) + 1  # +1 for overall
                n_cols = min(3, n_plots)
                n_rows = (n_plots + n_cols - 1) // n_cols
                
                fig = make_subplots(
                    rows=n_rows, cols=n_cols,
                    subplot_titles=[col.capitalize() for col in score_cols] + ['Overall'],
                    specs=[[{'type': 'bar'} for _ in range(n_cols)] for _ in range(n_rows)]
                )
                
                # Color mapping for response types
                color_map = {'Gold': 'gold', 'Anas': 'steelblue', 'LLM': 'crimson'}
                
                # Plot each criterion
                plot_idx = 0
                for criterion in score_cols:
                    if plot_idx >= n_rows * n_cols:
                        break
                        
                    row = plot_idx // n_cols + 1
                    col = plot_idx % n_cols + 1
                    
                    for response_type in selected_responses:
                        if response_type in filtered_data['response_type'].unique():
                            resp_data = filtered_data[filtered_data['response_type'] == response_type][criterion].dropna()
                            if len(resp_data) > 0:
                                # Round to handle floating point precision
                                resp_data_rounded = resp_data.round(1)
                                score_counts = resp_data_rounded.value_counts().reindex([0, 0.5, 1], fill_value=0)
                                
                                fig.add_trace(
                                    go.Bar(
                                        x=['0', '0.5', '1'],
                                        y=score_counts.values,
                                        name=response_type,
                                        marker_color=color_map.get(response_type, 'gray'),
                                        showlegend=(plot_idx == 0)
                                    ),
                                    row=row, col=col
                                )
                    
                    plot_idx += 1
                
                # Overall distribution in the last subplot
                if plot_idx < n_rows * n_cols:
                    row = plot_idx // n_cols + 1
                    col = plot_idx % n_cols + 1
                    
                    for response_type in selected_responses:
                        if response_type in filtered_data['response_type'].unique():
                            all_scores_resp = []
                            for col_name in score_cols:
                                col_scores = filtered_data[filtered_data['response_type'] == response_type][col_name].dropna().values
                                all_scores_resp.extend(col_scores)
                            
                            if all_scores_resp:
                                all_scores_array = np.array(all_scores_resp)
                                # Round to handle floating point precision
                                all_scores_rounded = np.round(all_scores_array, 1)
                                score_counts_resp = {}
                                for score in [0, 0.5, 1]:
                                    score_counts_resp[score] = (all_scores_rounded == score).sum()
                                
                                fig.add_trace(
                                    go.Bar(
                                        x=['0', '0.5', '1'],
                                        y=[score_counts_resp.get(0, 0), score_counts_resp.get(0.5, 0), score_counts_resp.get(1, 0)],
                                        name=response_type,
                                        marker_color=color_map.get(response_type, 'gray'),
                                        showlegend=False
                                    ),
                                    row=row, col=col
                                )
                
                fig.update_layout(
                    height=300 * n_rows,
                    showlegend=True,
                    title_text="Score Frequency Distribution by Criterion",
                    barmode='group',
                    legend_title="Response Type"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No score columns available for analysis.")
        
        with sub_tab2:
            # Average scores by criterion for each response type
            st.subheader("Average Scores by Criterion")
            
            if len(score_cols) > 0:
                summary_stats = filtered_data.groupby('response_type')[score_cols].mean()
                
                if not summary_stats.empty:
                    melted_data = summary_stats.reset_index().melt(
                        id_vars=['response_type'],
                        value_vars=score_cols,
                        var_name='criterion',
                        value_name='score'
                    )
                    
                    # Order criteria
                    criterion_order = score_cols
                    melted_data['criterion'] = pd.Categorical(melted_data['criterion'], categories=criterion_order, ordered=True)
                    melted_data = melted_data.sort_values('criterion')
                    
                    fig_criteria = px.bar(
                        melted_data,
                        x='criterion',
                        y='score',
                        color='response_type',
                        barmode='group',
                        color_discrete_map={'Gold': '#FFD700', 'Anas': '#2E86AB', 'LLM': '#A23B72'},
                        text_auto='.3f',
                        title='Average Score by Criterion and Response Type'
                    )
                    fig_criteria.update_layout(
                        height=500,
                        xaxis_title="Evaluation Criterion",
                        yaxis_title="Average Score",
                        legend_title="Response Type"
                    )
                    st.plotly_chart(fig_criteria, use_container_width=True)
                else:
                    st.info("No data available for criterion comparison.")
            else:
                st.info("No score columns available.")
        
        with sub_tab3:
            # Performance trends across cases
            st.subheader("Performance Trends Across Cases")
            
            if 'case_no' in filtered_data.columns and len(score_cols) > 0:
                case_performance = filtered_data.groupby(['case_no', 'response_type'])[score_cols].mean()
                if not case_performance.empty:
                    case_performance['average'] = case_performance.mean(axis=1)
                    case_performance_reset = case_performance.reset_index()
                    
                    # Sort cases
                    case_performance_reset = case_performance_reset.sort_values('case_no')
                    
                    fig_trends = px.line(
                        case_performance_reset,
                        x='case_no',
                        y='average',
                        color='response_type',
                        markers=True,
                        color_discrete_map={'Gold': '#FFD700', 'Anas': '#2E86AB', 'LLM': '#A23B72'},
                        title='Average Score by Case Number',
                        labels={'case_no': 'Case Number', 'average': 'Average Score'}
                    )
                    fig_trends.update_layout(
                        height=500,
                        xaxis=dict(tickmode='linear', dtick=1),
                        legend_title="Response Type"
                    )
                    st.plotly_chart(fig_trends, use_container_width=True)
                    
                    # Add summary table
                    st.subheader("Case Performance Summary")
                    
                    # Create a pivot table for easy viewing
                    pivot_table = case_performance_reset.pivot_table(
                        index='case_no',
                        columns='response_type',
                        values='average'
                    ).round(3)
                    
                    # Style the table
                    styled_table = pivot_table.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=1)
                    st.dataframe(styled_table, use_container_width=True)
                else:
                    st.info("No case performance data available.")
            else:
                st.info("Case number or score data not available.")

with tab3:
    st.subheader("👥 Annotator Agreement Analysis")
    
    if len(filtered_data) == 0:
        st.warning("No data available with current filters.")
    else:
        # Calculate exact agreement
        agreement_data = []
        
        for case in selected_cases:
            for response_type in selected_responses:
                case_data = filtered_data[
                    (filtered_data['case_no'] == case) & 
                    (filtered_data['response_type'] == response_type)
                ]
                
                if len(case_data) >= 2:
                    annotators_in_case = case_data['annotator'].unique()
                    
                    if len(annotators_in_case) >= 2:
                        scores_dict = {}
                        for annotator in annotators_in_case:
                            annotator_scores = case_data[case_data['annotator'] == annotator][score_cols].values
                            if len(annotator_scores) > 0:
                                scores_dict[annotator] = annotator_scores[0]
                        
                        if len(scores_dict) >= 2:
                            annotator_list = list(scores_dict.keys())
                            for i in range(len(annotator_list)):
                                for j in range(i+1, len(annotator_list)):
                                    ann1_scores = scores_dict[annotator_list[i]]
                                    ann2_scores = scores_dict[annotator_list[j]]
                                    
                                    # Convert to numeric arrays, handling NaN
                                    ann1_numeric = np.array([float(x) if pd.notna(x) else np.nan for x in ann1_scores])
                                    ann2_numeric = np.array([float(x) if pd.notna(x) else np.nan for x in ann2_scores])
                                    
                                    # Find valid comparisons (where both are not NaN)
                                    valid_mask = ~(np.isnan(ann1_numeric) | np.isnan(ann2_numeric))
                                    
                                    if valid_mask.any():
                                        exact_matches = (ann1_numeric[valid_mask] == ann2_numeric[valid_mask]).sum()
                                        agreement_percentage = (exact_matches / valid_mask.sum()) * 100
                                        
                                        agreement_data.append({
                                            'Case': int(case),
                                            'Response Type': response_type,
                                            'Annotator Pair': f"{annotator_list[i]}-{annotator_list[j]}",
                                            'Exact Agreement %': agreement_percentage,
                                            'Exact Matches': exact_matches,
                                            'Total Comparisons': valid_mask.sum()
                                        })
        
        if agreement_data:
            agreement_df = pd.DataFrame(agreement_data)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Visualize agreement
                fig_agreement = px.scatter(
                    agreement_df,
                    x='Case',
                    y='Exact Agreement %',
                    color='Response Type',
                    size='Exact Matches',
                    hover_data=['Annotator Pair', 'Total Comparisons'],
                    title='Exact Score Agreement Between Annotators',
                    color_discrete_map={'Gold': '#FFD700', 'Anas': '#2E86AB', 'LLM': '#A23B72'},
                    labels={'Exact Agreement %': 'Agreement (%)', 'Case': 'Case Number'}
                )
                fig_agreement.update_layout(
                    height=500,
                    xaxis=dict(tickmode='linear', dtick=1),
                    legend_title="Response Type"
                )
                fig_agreement.add_hline(y=100, line_dash="dash", line_color="green", annotation_text="Perfect Agreement")
                fig_agreement.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Good Agreement")
                fig_agreement.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Moderate Agreement")
                st.plotly_chart(fig_agreement, use_container_width=True)
            
            with col2:
                st.subheader("Agreement Statistics")
                
                # Summary statistics
                overall_agreement = agreement_df['Exact Agreement %'].mean()
                perfect_agreement = len(agreement_df[agreement_df['Exact Agreement %'] == 100])
                good_agreement = len(agreement_df[agreement_df['Exact Agreement %'] >= 80])
                moderate_agreement = len(agreement_df[agreement_df['Exact Agreement %'] >= 60])
                total_comparisons = len(agreement_df)
                
                st.metric("Overall Agreement", f"{overall_agreement:.1f}%")
                st.metric("Perfect Agreement (100%)", f"{perfect_agreement}/{total_comparisons}")
                st.metric("Good Agreement (≥80%)", f"{good_agreement}/{total_comparisons}")
                st.metric("Moderate Agreement (≥60%)", f"{moderate_agreement}/{total_comparisons}")
                
                # By response type
                st.subheader("By Response Type")
                type_agreement = agreement_df.groupby('Response Type')['Exact Agreement %'].mean().round(1)
                for rt, avg in type_agreement.items():
                    st.write(f"**{rt}**: {avg}%")
                
                # Show top agreement pairs
                st.subheader("Top Agreement Pairs")
                top_pairs = agreement_df.nlargest(5, 'Exact Agreement %')[['Annotator Pair', 'Exact Agreement %', 'Case']]
                st.dataframe(top_pairs, use_container_width=True)
        else:
            st.info("No agreement data available for the selected filters. Try adjusting filters or check if multiple annotators evaluated the same cases.")

with tab4:
    st.subheader("📋 Detailed Case Analysis")
    
    if len(filtered_data) == 0:
        st.warning("No data available with current filters.")
    else:
        # Create two columns for case selection and display
        col_left, col_right = st.columns([1, 3])
        
        with col_left:
            if len(selected_cases) > 0:
                selected_case = st.selectbox(
                    "Select Case:",
                    options=sorted(selected_cases),
                    format_func=lambda x: f"Case {int(x)}",
                    index=0
                )
                
                # Get case data
                case_data = filtered_data[filtered_data['case_no'] == selected_case]
                
                # Case summary
                st.metric("Total Evaluations", len(case_data))
                
                # Calculate case average
                if len(case_data) > 0 and len(score_cols) > 0:
                    numeric_scores = []
                    for col in score_cols:
                        if col in case_data.columns:
                            numeric_vals = pd.to_numeric(case_data[col], errors='coerce')
                            numeric_scores.extend(numeric_vals.dropna().values)
                    
                    if numeric_scores:
                        case_avg = np.mean(numeric_scores)
                        st.metric("Case Average Score", f"{case_avg:.3f}")
                    else:
                        st.metric("Case Average Score", "N/A")
                    
                    # Response type distribution
                    st.subheader("Response Types")
                    resp_counts = case_data['response_type'].value_counts()
                    for resp, count in resp_counts.items():
                        st.write(f"**{resp}**: {count}")
                else:
                    st.metric("Case Average Score", "N/A")
            else:
                st.info("No cases available.")
                selected_case = None
        
        with col_right:
            if selected_case is not None:
                case_data = filtered_data[filtered_data['case_no'] == selected_case]
                
                if len(case_data) > 0:
                    # Create a detailed table
                    st.subheader(f"Case {int(selected_case)} - Detailed Scores")
                    
                    # Prepare display data
                    display_data = []
                    for idx, row in case_data.iterrows():
                        row_dict = {
                            'Response Type': str(row.get('response_type', 'N/A')),
                            'Annotator': str(row.get('annotator', 'N/A'))
                        }
                        
                        # Add scores
                        scores = []
                        for col in score_cols:
                            if col in row:
                                score_val = row[col]
                                try:
                                    num_score = float(score_val)
                                    row_dict[col.capitalize()] = f"{num_score:.3f}"
                                    scores.append(num_score)
                                except (ValueError, TypeError):
                                    row_dict[col.capitalize()] = str(score_val)
                        
                        # Calculate average
                        if scores:
                            avg_score = np.mean(scores)
                            row_dict['Average'] = f"{avg_score:.3f}"
                        else:
                            row_dict['Average'] = 'N/A'
                        
                        display_data.append(row_dict)
                    
                    if display_data:
                        display_df = pd.DataFrame(display_data)
                        
                        # Display with highlighting
                        def highlight_scores(val):
                            try:
                                num_val = float(val)
                                if num_val == 1.0:
                                    return 'background-color: #10B981; color: white;'
                                elif num_val >= 0.5:
                                    return 'background-color: #F59E0B;'
                                elif num_val == 0.0:
                                    return 'background-color: #EF4444; color: white;'
                            except:
                                pass
                            return ''
                        
                        # Apply styling
                        styled_df = display_df.style.applymap(highlight_scores, subset=[col.capitalize() for col in score_cols] + ['Average'])
                        st.dataframe(styled_df, use_container_width=True, height=400)
                        
                        # Visual comparison
                        st.subheader("Score Comparison Visualization")
                        
                        # Prepare plot data
                        plot_data = []
                        for idx, row in case_data.iterrows():
                            response_type = str(row.get('response_type', 'N/A'))
                            annotator = str(row.get('annotator', 'N/A'))
                            
                            for col in score_cols:
                                if col in row:
                                    try:
                                        score_val = float(row[col])
                                        plot_data.append({
                                            'Response': response_type,
                                            'Annotator': annotator,
                                            'Criterion': col.capitalize(),
                                            'Score': score_val,
                                            'Label': f"{response_type} - {annotator}"
                                        })
                                    except (ValueError, TypeError):
                                        continue
                        
                        if plot_data:
                            plot_df = pd.DataFrame(plot_data)
                            
                            # Create interactive plot
                            fig_comparison = px.scatter(
                                plot_df,
                                x='Criterion',
                                y='Score',
                                color='Response',
                                symbol='Annotator',
                                hover_data=['Label'],
                                title=f'Case {int(selected_case)}: Score Comparison',
                                color_discrete_map={'Gold': '#FFD700', 'Anas': '#2E86AB', 'LLM': '#A23B72'}
                            )
                            fig_comparison.update_layout(
                                height=400,
                                yaxis=dict(range=[-0.1, 1.1], tickvals=[0, 0.5, 1]),
                                xaxis_title="Evaluation Criterion",
                                yaxis_title="Score",
                                legend_title="Response Type / Annotator"
                            )
                            st.plotly_chart(fig_comparison, use_container_width=True)
                        else:
                            st.info("No valid score data for visualization.")
                    else:
                        st.info("No data to display for this case.")
                else:
                    st.info("No data available for the selected case.")
            else:
                st.info("Please select a case from the left panel.")

with tab5:
    st.subheader("📊 Heatmap Analysis")
    
    if len(filtered_data) == 0:
        st.warning("No data available with current filters.")
    else:
        # Prepare heatmap data for agreement
        heatmap_data = []
        
        for case in selected_cases:
            for response_type in selected_responses:
                case_responses = filtered_data[
                    (filtered_data['case_no'] == case) & 
                    (filtered_data['response_type'] == response_type)
                ]
                
                if len(case_responses) >= 2:
                    annotators_present = case_responses['annotator'].unique()
                    
                    if len(annotators_present) >= 2:
                        scores_list = []
                        for annotator in annotators_present:
                            scores = case_responses[case_responses['annotator'] == annotator][score_cols].values
                            if len(scores) > 0:
                                scores_list.append(scores[0])
                        
                        if len(scores_list) >= 2:
                            total_agreement = 0
                            pair_count = 0
                            
                            for i in range(len(scores_list)):
                                for j in range(i+1, len(scores_list)):
                                    # Convert to numeric arrays
                                    ann1_scores = np.array([float(x) if pd.notna(x) else np.nan for x in scores_list[i]])
                                    ann2_scores = np.array([float(x) if pd.notna(x) else np.nan for x in scores_list[j]])
                                    
                                    # Find valid comparisons
                                    valid_mask = ~(np.isnan(ann1_scores) | np.isnan(ann2_scores))
                                    
                                    if valid_mask.any():
                                        agreement = (ann1_scores[valid_mask] == ann2_scores[valid_mask]).sum() / valid_mask.sum() * 100
                                        total_agreement += agreement
                                        pair_count += 1
                            
                            avg_agreement = total_agreement / pair_count if pair_count > 0 else 0
                            
                            heatmap_data.append({
                                'Case': f'Case {int(case)}',
                                'Response Type': response_type,
                                'Avg Agreement %': avg_agreement
                            })
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            
            # Agreement heatmap
            st.subheader("Agreement Heatmap")
            
            pivot_heatmap = heatmap_df.pivot_table(
                values='Avg Agreement %',
                index='Case',
                columns='Response Type',
                aggfunc='mean'
            ).fillna(0)
            
            # Sort cases numerically
            pivot_heatmap.index = pd.Categorical(
                pivot_heatmap.index,
                categories=sorted(pivot_heatmap.index, key=lambda x: int(x.split()[1])),
                ordered=True
            )
            pivot_heatmap = pivot_heatmap.sort_index()
            
            fig_agreement_heatmap = px.imshow(
                pivot_heatmap.values,
                x=pivot_heatmap.columns,
                y=pivot_heatmap.index,
                color_continuous_scale='RdYlGn',
                aspect='auto',
                text_auto='.0f',
                title='Average Annotator Agreement Percentage',
                labels=dict(x="Response Type", y="Case", color="Agreement %"),
                zmin=0,
                zmax=100
            )
            fig_agreement_heatmap.update_layout(height=500)
            st.plotly_chart(fig_agreement_heatmap, use_container_width=True)
            
            # Performance heatmap
            st.subheader("Performance Heatmap")
            
            # Calculate average scores by case and response type
            if 'case_no' in filtered_data.columns and len(score_cols) > 0:
                performance_data = filtered_data.groupby(['case_no', 'response_type'])[score_cols].mean()
                if not performance_data.empty:
                    performance_data['average'] = performance_data.mean(axis=1)
                    performance_pivot = performance_data.reset_index().pivot_table(
                        values='average',
                        index='case_no',
                        columns='response_type'
                    )
                    
                    # Sort cases
                    performance_pivot = performance_pivot.sort_index()
                    
                    fig_perf_heatmap = px.imshow(
                        performance_pivot.values,
                        x=performance_pivot.columns,
                        y=[f'Case {int(i)}' for i in performance_pivot.index],
                        color_continuous_scale='RdYlGn',
                        aspect='auto',
                        text_auto='.3f',
                        title='Average Scores by Case and Response Type',
                        labels=dict(x="Response Type", y="Case", color="Score"),
                        zmin=0,
                        zmax=1
                    )
                    fig_perf_heatmap.update_layout(height=500)
                    st.plotly_chart(fig_perf_heatmap, use_container_width=True)
                else:
                    st.info("No performance data available for heatmap.")
            else:
                st.info("Case number or score data not available for heatmap.")
        else:
            st.info("No heatmap data available for the selected filters.")

# Footer with dataset summary
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
    <h4>📊 Dataset Summary</h4>
    <p><strong>{len(selected_cases)} of 12 Cases</strong> | <strong>{len(selected_responses)} Response Types</strong> | <strong>{len(selected_annotators)} Annotators</strong> | <strong>{len(score_cols)} Evaluation Criteria</strong></p>
    <p><strong>Response Types:</strong> {' • '.join(selected_responses)}</p> 
    <p><strong>Evaluation Criteria:</strong> {' • '.join([col.capitalize() for col in score_cols])}</p>
    <p><strong>Scoring Scale:</strong> 0.0 (Poor) • 0.5 (Average) • 1.0 (Excellent)</p>
    <p><strong>Total Data Points:</strong> {len(filtered_data):,} evaluations, {filtered_data[score_cols].notna().sum().sum():,} scores</p>
</div>
""", unsafe_allow_html=True)

# Add timestamp
st.markdown(f"*Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# Add download button for filtered data
st.sidebar.markdown("---")
if st.sidebar.button("📥 Download Filtered Data"):
    csv = filtered_data.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"llm_evaluation_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Add reset filters button
if st.sidebar.button("🔄 Reset All Filters"):
    st.rerun()

