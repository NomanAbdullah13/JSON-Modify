import json
import time
import streamlit as st
from chatbot_setup import process_records_parallel, gpt_translate_text, QuotaExceededException

st.set_page_config(page_title="Q/A Validator", layout="wide")
st.title("JSON Question-Answer Validator & Fixer")

# Initialize session state
for key in ["original_data", "processed_data", "processed_en_data", "last_uploaded_file", "quota_exceeded"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Function to display quota exceeded message
def display_quota_exceeded_message():
    st.error("🚨 **QUOTA EXCEEDED** 🚨")
    st.markdown("""
    ### ❌ OpenAI API Quota Limit Reached
    
    **Your OpenAI API usage limit has been exceeded.**
    
    #### 🔧 **Immediate Actions Required:**
    1. **Check your OpenAI account billing** at [OpenAI Platform](https://platform.openai.com/account/billing)
    2. **Add credits** to your OpenAI account 
    3. **Verify your payment method** is valid and active
    4. **Check your usage limits** in your OpenAI dashboard
    
    #### 💡 **Next Steps:**
    - Visit [OpenAI Platform](https://platform.openai.com/) to manage your account
    - Consider upgrading to a higher tier plan for more credits
    - Monitor your API usage to avoid future interruptions
    
    #### ⏳ **Processing Status:**
    - **Processing has been stopped** to prevent further quota usage
    - **Partial results** (if any) are preserved below
    - **Restart processing** after resolving billing issues
    """)
    
    # Add a reset button
    if st.button("🔄 Reset Session (After Fixing Billing)", type="primary"):
        for key in st.session_state.keys():
            if key not in ["original_data", "last_uploaded_file"]:
                st.session_state[key] = None
        st.rerun()

uploaded = st.file_uploader("Upload JSON File", type="json")

# Only load data on NEW upload (check if it's a different file)
if uploaded and uploaded.name != st.session_state.get("last_uploaded_file"):
    try:
        st.session_state["original_data"] = json.load(uploaded)
        if isinstance(st.session_state["original_data"], dict):
            st.session_state["original_data"] = [st.session_state["original_data"]]
        # Reset processed data only on new upload
        st.session_state["processed_data"] = None
        st.session_state["processed_en_data"] = None
        st.session_state["quota_exceeded"] = None
        st.session_state["last_uploaded_file"] = uploaded.name
        st.success(f"✅ Successfully loaded {len(st.session_state['original_data'])} records from {uploaded.name}")
    except json.JSONDecodeError as e:
        st.error(f"❌ Invalid JSON file: {str(e)}")
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

# Check if quota was exceeded - show message and stop execution
if st.session_state.get("quota_exceeded"):
    display_quota_exceeded_message()
    st.stop()  # Stop Streamlit execution here

# Show original data
if st.session_state["original_data"]:
    st.subheader("📄 Original Data")
    with st.expander("View Original Data", expanded=False):
        st.json(st.session_state["original_data"])

# Processing
if st.button("🚀 Start Processing", disabled=not st.session_state.get("original_data")):
    # Only process if not already processed
    if not st.session_state["processed_data"]:
        st.info("🔄 Starting validation and correction process...")
        start_time = time.time()
        total = len(st.session_state["original_data"])
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Processing with quota exception handling
            processed = []
            batch_size = 5
            for i in range(0, total, batch_size):
                batch = st.session_state["original_data"][i:i+batch_size]
                status_text.text(f"Processing batch {i//batch_size + 1}...")
                
                try:
                    batch_results = process_records_parallel(batch, max_workers=batch_size)
                    processed.extend(batch_results)
                    progress_bar.progress(min(100, int((len(processed)/total)*100)))
                    status_text.text(f"Processed {len(processed)} of {total} records...")
                except QuotaExceededException as e:
                    st.session_state["quota_exceeded"] = str(e)
                    st.session_state["processed_data"] = processed  # Save partial results
                    st.rerun()  # Reload page to show quota message

            elapsed = time.time() - start_time
            st.session_state["processed_data"] = processed
            status_text.text("✅ Processing completed!")

            # Translation phase
            st.info("🌐 Starting translation to English...")
            translation_progress = st.progress(0)
            translation_status = st.empty()
            
            try:
                processed_en = []
                for i, rec in enumerate(processed):
                    translation_status.text(f"Translating record {i+1} of {len(processed)}...")
                    try:
                        translated_rec = gpt_translate_text(rec, "English")
                        processed_en.append(translated_rec)
                        translation_progress.progress(int(((i+1)/len(processed))*100))
                    except QuotaExceededException as e:
                        st.session_state["quota_exceeded"] = str(e)
                        st.session_state["processed_en_data"] = processed_en  # Save partial translations
                        st.rerun()  # Reload page to show quota message
                
                st.session_state["processed_en_data"] = processed_en
                translation_status.text("✅ Translation completed!")
                
            except QuotaExceededException as e:
                st.session_state["quota_exceeded"] = str(e)
                st.rerun()  # Reload page to show quota message

            st.success(f"✅ Processing completed in {elapsed:.2f} seconds!")
            
        except QuotaExceededException as e:
            st.session_state["quota_exceeded"] = str(e)
            st.rerun()  # Reload page to show quota message
            
    else:
        st.info("✅ Data already processed. Results are shown below.")

# Show processed data if exists
if st.session_state["processed_data"]:
    st.subheader("✅ Processed Data")
    with st.expander("View Processed Data", expanded=False):
        st.json(st.session_state["processed_data"])

    # Summary
    processed = st.session_state["processed_data"]
    valid = sum(1 for r in processed if r.get("_validation", {}).get("valid") is True)
    corrected = sum(1 for r in processed if r.get("_validation", {}).get("corrected") is True)
    failed = sum(1 for r in processed if not r.get("_validation", {}).get("valid", False) and not r.get("_validation", {}).get("corrected", False))
    
    st.subheader("📊 Processing Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(processed))
    with col2:
        st.metric("Valid (No Changes)", valid, delta=None)
    with col3:
        st.metric("Corrected by AI", corrected, delta=None)
    with col4:
        st.metric("Failed/Flagged", failed, delta=None if failed == 0 else f"{failed}")

    # Download buttons
    st.subheader("📥 Download Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "📄 Download Corrected JSON (Original Language)",
            data=json.dumps(st.session_state["processed_data"], indent=2, ensure_ascii=False),
            file_name="corrected.json",
            mime="application/json",
            help="Download the corrected data in the original language"
        )
    
    with col2:
        if st.session_state.get("processed_en_data"):
            st.download_button(
                "🌐 Download Corrected JSON (English)",
                data=json.dumps(st.session_state["processed_en_data"], indent=2, ensure_ascii=False),
                file_name="corrected_en.json",
                mime="application/json",
                help="Download the corrected data translated to English"
            )
        else:
            st.button("🌐 English Translation Unavailable", disabled=True, 
                     help="Translation not completed or failed due to quota limits")

# Instructions
if not st.session_state.get("original_data"):
    st.subheader("📋 Instructions")
    st.write("""
    1. **Upload** a JSON file containing question-answer records
    2. **Review** the original data structure  
    3. **Click** 'Start Processing' to validate and correct answers using AI
    4. **Download** the corrected results in original language or English
    
    **⚠️ Important Notes:**
    - Make sure your OpenAI API key is configured in the `.env` file
    - Ensure you have sufficient OpenAI credits for processing
    - Processing will stop if quota limits are reached
    """)

# Footer with API and quota information
st.markdown("---")
st.markdown("### 🆘 Important Note")

col1, col2 = st.columns(2)

with col1:
    with st.expander("🔑 API Key & Authentication Issues"):
        st.write("""
        **API Key Problems:**
        - Ensure your `.env` file contains: `OPENAI_API_KEY=your_api_key_here`
        - Verify the API key is valid and active at [OpenAI Platform](https://platform.openai.com/account/api-keys)
        - Make sure the API key hasn't been revoked or expired
        
        **Authentication Errors:**
        - Double-check the API key format (starts with `sk-`)
        - Ensure no extra spaces or characters in the key
        - Try regenerating your API key if issues persist
        
        **Rate Limit Issues:**
        - Wait a few minutes between processing batches
        - The app automatically retries after rate limit delays
        - Consider processing smaller files to avoid limits
        
        **Connection Problems:**
        - Check your internet connection
        - Verify firewall settings allow OpenAI API access
        - Try again after a few moments if experiencing network issues
        """)

with col2:
    with st.expander("💳 Quota & Billing Management"):
        st.write("""
        **What are API Quotas?**
        - OpenAI limits API usage based on your plan and available credits
        - Each API call consumes credits from your account
        - Processing stops when limits are reached to prevent unexpected charges
        
        **Managing Your Quota:**
        - Monitor usage at [OpenAI Platform](https://platform.openai.com/account/usage)
        - Add credits at [OpenAI Billing](https://platform.openai.com/account/billing)
        - Consider upgrading plans for higher limits
        
        **If Quota Exceeded:**
        - Check your OpenAI account billing status
        - Add more credits to continue processing
        - Verify your payment method is valid and active
        - Use the 'Reset Session' button after fixing billing issues
        
        **Cost Management:**
        - Monitor your spending in the OpenAI dashboard
        - Set usage alerts to avoid unexpected charges
        - Consider processing in smaller batches to control costs
        """)