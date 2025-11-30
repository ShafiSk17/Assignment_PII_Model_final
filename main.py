import streamlit as st
import pandas as pd
import io
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from typing import Dict, List, Tuple
import re
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

# Page config
st.set_page_config(
    page_title="🚀 Enterprise PII Detection Demo",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; font-weight: bold;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 1rem; border-radius: 10px;}
    .success-box {background-color: #d4edda; padding: 1rem; border-radius: 10px; border-left: 5px solid #28a745;}
    .warning-box {background-color: #fff3cd; padding: 1rem; border-radius: 10px; border-left: 5px solid #ffc107;}
</style>
""", unsafe_allow_html=True)

class PIIDetector:
    def __init__(self):
        self.analyzer = self._setup_analyzer()
        self.anonymizer = AnonymizerEngine()
        
    def _setup_analyzer(self):
        """Setup Presidio analyzer with spaCy NLP engine"""
        # Configure spaCy NLP engine
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_lg"}  # English large model
            ]
        }

        provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
        nlp_engine = provider.create_engine()

        # Initialize analyzer
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

        # Predefined entities are already available in Presidio
        return analyzer

    def add_custom_regex(self, name: str, pattern: str, score: float = 0.8) -> None:
        """Dynamically add custom regex recognizer"""
        recognizer = {
            "name": name,
            "supported_entity": name,
            "supported_language": "en",
            "deny_list": [],
            "patterns": [{"regex": pattern, "score": score}]
        }
        self.analyzer.registry.add_recognizer(recognizer)
    
    def add_custom_whitelist(self, name: str, whitelist: List[str], score: float = 0.8) -> None:
        """Dynamically add custom whitelist recognizer"""
        recognizer = {
            "name": name,
            "supported_entity": name,
            "supported_language": "en",
            "deny_list": [],
            "context": [],
            "patterns": [],
            "whitelist_patterns": [{"words": whitelist, "score": score}]
        }
        self.analyzer.registry.add_recognizer(recognizer)

# Initialize detector
detector = PIIDetector()

# Main App
def main():
    st.markdown('<h1 class="main-header">🔒 Enterprise PII Detection & Classification</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for custom rules
    with st.sidebar:
        st.header("⚙️ Custom PII Rules")
        
        # Custom Whitelist Section
        st.subheader("📝 Custom Whitelist")
        whitelist_name = st.text_input("Tag Name", value="SCHOOL_ID", key="whitelist_name")
        whitelist_examples = st.text_area(
            "Examples (one per line)", 
            value="223j1ao5g\nschool123\nstudent456", 
            key="whitelist_examples",
            help="Enter examples of your custom identifier"
        )
        
        # Custom Regex Section
        st.subheader("🔤 Custom Regex")
        regex_name = st.text_input("Tag Name", value="CUSTOM_ID", key="regex_name")
        regex_pattern = st.text_input(
            "Regex Pattern", 
            value=r"^[A-Za-z0-9]{8,12}$", 
            key="regex_pattern",
            help="Enter regex pattern for detection"
        )
        
        # Add rules buttons
        col1, col2 = st.columns(2)
        if col1.button("➕ Add Whitelist Rule", type="primary"):
            if whitelist_name and whitelist_examples.strip():
                examples = [x.strip() for x in whitelist_examples.strip().split('\n') if x.strip()]
                detector.add_custom_whitelist(whitelist_name, examples)
                st.success(f"✅ Added {whitelist_name} whitelist rule!")
            else:
                st.error("Please enter tag name and examples")
        
        if col2.button("➕ Add Regex Rule"):
            if regex_name and regex_pattern:
                detector.add_custom_regex(regex_name, regex_pattern)
                st.success(f"✅ Added {regex_name} regex rule!")
            else:
                st.error("Please enter tag name and regex pattern")
        
        st.markdown("---")
        st.info("**Predefined Presidio Entities (Auto-enabled):**\n\n" +
                "• PERSON • PHONE_NUMBER • EMAIL_ADDRESS • CREDIT_CARD\n" +
                "• US_SSN • URL • IP_ADDRESS • DOMAIN_NAME\n" +
                "• USERNAME • US_PASSPORT • MEDICAL_LICENSE")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload Document")
        
        # File uploader with format selection
        file_type = st.radio("Select file type:", 
                           ["📄 CSV", "📄 PDF", "📄 Text", "📄 JSON"], 
                           horizontal=True)
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=[file_type.lower().replace('📄 ', '')],
            help="Upload your document for PII scanning"
        )
    
    with col2:
        st.header("🎯 Quick Test")
        quick_text = st.text_area("Or paste text directly:", height=150, 
                                placeholder="Enter text to test PII detection...")
        
        if st.button("🚀 Quick Scan", type="secondary"):
            if quick_text:
                results = analyze_text(quick_text)
                display_results(quick_text, results)
    
    # Process uploaded file
    if uploaded_file is not None:
        with st.spinner("🔍 Scanning for PII..."):
            content, filename = load_file(uploaded_file, file_type)
            
            if isinstance(content, pd.DataFrame):
                # Handle CSV/DataFrame
                results_df = analyze_dataframe(content)
                st.subheader(f"📊 Results for {filename}")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total PII Found", len(results_df), delta="+5")
                    st.dataframe(results_df, use_container_width=True)
                
                with col2:
                    if not results_df.empty:
                        safe_df = anonymize_dataframe(content, results_df)
                        csv_buffer = io.StringIO()
                        safe_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            "⬇️ Download Safe CSV",
                            csv_buffer.getvalue(),
                            f"safe_{filename}",
                            "text/csv"
                        )
                    st.success("✅ PII detected and ready for anonymization!")
            
            elif isinstance(content, str):
                # Handle text/PDF content
                results = analyze_text(content)
                st.subheader(f"📄 Results for {filename}")
                display_results(content, results)
                
                # Anonymized version
                anonymized = anonymize_text(content, results)
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("🔒 Anonymized Version:", anonymized, height=300)
                with col2:
                    st.download_button(
                        "⬇️ Download Safe Version",
                        anonymized,
                        f"safe_{filename}",
                        "text/plain"
                    )

def load_file(uploaded_file, file_type):
    """Load different file formats"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        if "csv" in file_type.lower():
            content = pd.read_csv(tmp_path)
        elif "pdf" in file_type.lower():
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            content = " ".join([doc.page_content for doc in docs])
        elif "json" in file_type.lower():
            # Directly read structured JSON as DataFrame
            content = pd.read_json(tmp_path)
        else:
            with open(tmp_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        return content, uploaded_file.name
    finally:
        os.unlink(tmp_path)

def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze DataFrame for PII across all columns"""
    results = []
    
    for col in df.columns:
        for idx, value in enumerate(df[col].dropna()):
            if pd.isna(value):
                continue
                
            text = str(value)
            findings = detector.analyzer.analyze(text=text, entities=[], language='en')
            
            for finding in findings:
                results.append({
                    'Column': col,
                    'Row': idx,
                    'PII_Type': finding.entity_type,
                    'Text': text[finding.start:finding.end],  # <-- FIXED
                    'Score': finding.score,
                    'Start': finding.start,
                    'End': finding.end
                })
    
    return pd.DataFrame(results)


def analyze_text(text: str) -> List:
    """Analyze text for PII"""
    return detector.analyzer.analyze(text=text, entities=[], language='en')

def anonymize_dataframe(df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """✅ FIXED: Anonymize DataFrame using proper RecognizerResult objects"""
    df_safe = df.copy()
    
    for _, row in results_df.iterrows():
        col = row['Column']
        idx = row['Row']
        start, end = row['Start'], row['End']
        pii_type = row['PII_Type']
        
        original_text = str(df_safe.iloc[idx][col])
        
        # ✅ FIX: Create proper RecognizerResult object (NOT dict)
        recognizer_result = RecognizerResult(
            entity_type=pii_type,
            start=start,
            end=end,
            score=row['Score'],
            analysis_explanation="Detected by Presidio"
        )
        
        # Now anonymize with proper object
        redacted = detector.anonymizer.anonymize(
            text=original_text, 
            analyzer_results=[recognizer_result]
        )
        
        df_safe.iloc[idx, df.columns.get_loc(col)] = redacted.text
    
    return df_safe

def anonymize_text(text: str, findings: List) -> str:
    """Anonymize text based on findings"""
    return detector.anonymizer.anonymize(text=text, analyzer_results=findings).text

def display_results(original_text: str, findings: List):
    """Display results with highlighting"""
    if findings:
        st.success(f"✅ Found **{len(findings)}** PII entities!")

        # Build results table
        results_df = pd.DataFrame([{
            'PII Type': f.entity_type,
            'Text': original_text[f.start:f.end],  # slice text from original
            'Confidence': f"{f.score:.2f}",
            'Location': f"{f.start}-{f.end}"
        } for f in findings])

        st.dataframe(results_df, use_container_width=True)

        # Highlighted text
        highlighted = original_text
        for f in reversed(findings):  # Reverse to maintain positions
            highlighted = (
                highlighted[:f.start] +
                f"**[PII:{f.entity_type}]**" +
                highlighted[f.end:]
            )

        st.markdown("**📋 Highlighted Text:**")
        st.markdown(highlighted)
    else:
        st.success("🎉 No PII detected! Safe to use.")


if __name__ == "__main__":
    main()
