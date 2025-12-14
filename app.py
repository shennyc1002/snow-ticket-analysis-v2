"""Streamlit Web UI for ServiceNow Ticket Analysis Agent."""

import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path
from io import BytesIO

from excel_handler import ExcelHandler, ExcelValidationError
from categorization_engine import CategorizationEngine, CategorizationError
from similarity_detector import SimilarityDetector
from validator import DataValidator, ValidationError
import config

# Page configuration
st.set_page_config(
    page_title="ServiceNow Ticket Analyzer",
    page_icon="🎫",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'categories_summary' not in st.session_state:
        st.session_state.categories_summary = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = None


def validate_api_key(api_key: str) -> bool:
    """Check if the API key is valid format."""
    return api_key and len(api_key) > 20 and api_key.startswith('sk-')


def process_tickets(df: pd.DataFrame, progress_bar, status_text) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process tickets and return categorized DataFrame and summary."""

    # Reset DataFrame index to ensure consistent indexing
    df = df.reset_index(drop=True)

    # Create handler with DataFrame
    excel_handler = ExcelHandler.__new__(ExcelHandler)
    excel_handler.df = df.copy()
    excel_handler.original_row_count = len(df)

    # Initialize components
    categorization_engine = CategorizationEngine()
    similarity_detector = SimilarityDetector()
    validator = DataValidator(df)

    # Extract ticket data
    status_text.text("Extracting ticket data...")
    tickets = excel_handler.get_ticket_data()
    progress_bar.progress(10)

    status_text.text(f"Found {len(tickets)} tickets to categorize...")

    # Categorize in batches
    status_text.text("Categorizing tickets with GPT...")
    all_categories = {}
    batch_size = config.BATCH_SIZE
    batches = [tickets[i:i + batch_size] for i in range(0, len(tickets), batch_size)]

    for i, batch in enumerate(batches):
        try:
            existing_cats = categorization_engine.get_established_categories()
            batch_categories = categorization_engine.categorize_batch(batch, existing_cats)
            all_categories.update(batch_categories)
        except Exception as e:
            # Retry failed batch one ticket at a time
            status_text.text(f"Batch {i+1} failed, retrying individually...")
            for ticket in batch:
                if ticket['index'] not in all_categories:
                    try:
                        single_result = categorization_engine.categorize_batch(
                            [ticket],
                            categorization_engine.get_established_categories()
                        )
                        all_categories.update(single_result)
                    except Exception:
                        # Assign default category for failed tickets
                        all_categories[ticket['index']] = "Uncategorized"

        progress = 10 + int((i + 1) / len(batches) * 50)
        progress_bar.progress(progress)
        status_text.text(f"Categorizing tickets... Batch {i+1}/{len(batches)} ({len(all_categories)}/{len(tickets)} done)")

    # Apply similarity consistency
    status_text.text("Applying similarity-based consistency...")
    progress_bar.progress(65)
    consistent_categories = similarity_detector.enforce_category_consistency(
        all_categories, tickets
    )

    # Normalize categories
    status_text.text("Normalizing category names...")
    progress_bar.progress(75)
    consistent_categories = categorization_engine.validate_and_normalize_categories(
        consistent_categories
    )

    # Validate
    status_text.text("Validating results...")
    progress_bar.progress(85)
    validator.validate_categories(consistent_categories)

    # Add categories to DataFrame
    excel_handler.add_categories(consistent_categories)
    validator.validate_final_output(excel_handler.df)

    progress_bar.progress(95)

    # Generate summary
    status_text.text("Generating summary...")
    summary = excel_handler.generate_summary()

    progress_bar.progress(100)
    status_text.text("Complete!")

    return excel_handler.df, summary


def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to Excel bytes for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Categorized Tickets')
    return output.getvalue()


def main():
    initialize_session_state()

    # Header
    st.markdown('<p class="main-header">🎫 ServiceNow Ticket Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your ServiceNow export and get intelligent ticket categorization powered by GPT</p>', unsafe_allow_html=True)

    # Sidebar for configuration
    with st.sidebar:
        st.header("API Configuration")

        # API Type Selection
        api_type = st.radio(
            "API Type",
            options=["Organization Wrapper (OAuth2)", "Direct OpenAI"],
            index=0 if config.USE_WRAPPER_API else 1,
            help="Choose how to connect to GPT"
        )

        use_wrapper = api_type == "Organization Wrapper (OAuth2)"

        if use_wrapper:
            st.subheader("OAuth2 Settings")

            auth_url = st.text_input(
                "Auth URL",
                value=os.getenv("AUTH_URL", config.AUTH_URL),
                help="OAuth2 token endpoint URL"
            )

            client_id = st.text_input(
                "Client ID",
                value=os.getenv("CLIENT_ID", ""),
                help="OAuth2 client ID"
            )

            client_secret = st.text_input(
                "Client Secret",
                type="password",
                value=os.getenv("CLIENT_SECRET", ""),
                help="OAuth2 client secret"
            )

            gpt_wrapper_url = st.text_input(
                "GPT Wrapper URL",
                value=os.getenv("GPT_WRAPPER_URL", config.GPT_WRAPPER_URL),
                help="URL for the GPT wrapper API"
            )

            # Update config
            if auth_url and client_id and client_secret and gpt_wrapper_url:
                config.AUTH_URL = auth_url
                config.CLIENT_ID = client_id
                config.CLIENT_SECRET = client_secret
                config.GPT_WRAPPER_URL = gpt_wrapper_url
                config.USE_WRAPPER_API = True
                os.environ["AUTH_URL"] = auth_url
                os.environ["CLIENT_ID"] = client_id
                os.environ["CLIENT_SECRET"] = client_secret
                os.environ["GPT_WRAPPER_URL"] = gpt_wrapper_url
                api_configured = True
            else:
                api_configured = False

        else:
            st.subheader("OpenAI Settings")

            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key (starts with 'sk-')",
                value=os.getenv("OPENAI_API_KEY", "")
            )

            if api_key:
                config.OPENAI_API_KEY = api_key
                config.USE_WRAPPER_API = False
                os.environ["OPENAI_API_KEY"] = api_key
                api_configured = True
            else:
                api_configured = False

        st.divider()
        st.header("Model Settings")

        # Model selection
        model = st.selectbox(
            "GPT Model",
            options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            help="gpt-4o: Best quality | gpt-4o-mini: Faster & cheaper"
        )
        config.MODEL_NAME = model

        # Batch size
        batch_size = st.slider(
            "Batch Size",
            min_value=5,
            max_value=50,
            value=20,
            help="Number of tickets processed per API call"
        )
        config.BATCH_SIZE = batch_size

        # Max categories
        max_cats = st.slider(
            "Max Categories",
            min_value=10,
            max_value=50,
            value=25,
            help="Maximum number of categories to create"
        )
        config.MAX_CATEGORIES = max_cats

        st.divider()
        st.markdown("**Required Columns:**")
        st.markdown("- Short Description")
        st.markdown("- Description")
        st.markdown("- Solution")
        st.markdown("- Priority")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📤 Upload Excel File")
        uploaded_file = st.file_uploader(
            "Choose your ServiceNow export file",
            type=['xlsx', 'xls'],
            help="Upload an Excel file exported from ServiceNow"
        )

    with col2:
        st.header("📊 Quick Stats")
        if uploaded_file:
            try:
                # Check if we need to reload (new file or first load)
                file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                if 'file_key' not in st.session_state or st.session_state.file_key != file_key:
                    # Reset file pointer and read all data
                    uploaded_file.seek(0)
                    df = pd.read_excel(uploaded_file)
                    # Store in session state
                    st.session_state.uploaded_df = df
                    st.session_state.file_key = file_key
                    # Reset processing state for new file
                    st.session_state.processing_complete = False
                    st.session_state.processed_df = None

                df = st.session_state.uploaded_df
                st.metric("Total Tickets", len(df))
                st.metric("Columns Found", len(df.columns))

                # Check required columns
                required = ['Short Description', 'Description', 'Solution', 'Priority']
                missing = [col for col in required if col not in df.columns]
                if missing:
                    st.error(f"Missing: {', '.join(missing)}")
                else:
                    st.success("All required columns found!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # Preview section
    if uploaded_file and 'uploaded_df' in st.session_state and st.session_state.uploaded_df is not None:
        st.divider()
        st.header("👀 Data Preview")

        try:
            df = st.session_state.uploaded_df
            st.info(f"Loaded {len(df)} rows from Excel file")
            st.dataframe(df.head(10), use_container_width=True)

            # Process button
            st.divider()

            if not api_configured:
                st.warning("Please configure API credentials in the sidebar to proceed.")
            else:
                if st.button("🚀 Analyze & Categorize Tickets", type="primary", use_container_width=True):

                    # Check required columns
                    required = ['Short Description', 'Description', 'Solution', 'Priority']
                    missing = [col for col in required if col not in df.columns]

                    if missing:
                        st.error(f"Missing required columns: {', '.join(missing)}")
                    else:
                        # Process tickets
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        try:
                            with st.spinner("Processing..."):
                                processed_df, summary = process_tickets(df, progress_bar, status_text)

                            st.session_state.processed_df = processed_df
                            st.session_state.categories_summary = summary
                            st.session_state.processing_complete = True

                            st.success("✅ Analysis complete!")

                        except (CategorizationError, ValidationError, ExcelValidationError) as e:
                            st.error(f"❌ Error: {e}")
                        except Exception as e:
                            st.error(f"❌ Unexpected error: {e}")
                            st.exception(e)

        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Results section
    if st.session_state.processing_complete and st.session_state.processed_df is not None:
        st.divider()
        st.header("📈 Results")

        # Summary metrics
        df = st.session_state.processed_df
        summary = st.session_state.categories_summary

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tickets", len(df))
        with col2:
            st.metric("Categories Created", df['Category'].nunique())
        with col3:
            top_category = df['Category'].value_counts().index[0]
            st.metric("Top Category", top_category)
        with col4:
            top_count = df['Category'].value_counts().iloc[0]
            st.metric("Top Category Count", top_count)

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Categorized Data", "📊 Category Summary", "📉 Visualization"])

        with tab1:
            st.dataframe(df, use_container_width=True, height=400)

        with tab2:
            st.dataframe(summary, use_container_width=True)

        with tab3:
            # Bar chart of categories
            category_counts = df['Category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            st.bar_chart(category_counts.set_index('Category'))

        # Download section
        st.divider()
        st.header("💾 Download Results")

        col1, col2 = st.columns(2)

        with col1:
            excel_data = convert_df_to_excel(df)
            st.download_button(
                label="📥 Download Categorized Excel",
                data=excel_data,
                file_name="snow_tickets_categorized.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with col2:
            summary_excel = convert_df_to_excel(summary.reset_index())
            st.download_button(
                label="📥 Download Summary Excel",
                data=summary_excel,
                file_name="category_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
