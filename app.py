#!/usr/bin/env python3
"""
app.py - Interface Streamlit Comparative
Démontre la différence entre LLM pur (hallucinations) et RAG+Patterns
"""

import streamlit as st
import time

# Configuration de la page
st.set_page_config(
    page_title="jQuery → JS Translator | Étude Comparative",
    layout="wide"
)

# Import des deux systèmes
try:
    from translator import SecurejQueryToJSTranslator, SecurityLevel
    RAG_SYSTEM_OK = True
except ImportError as e:
    st.error(f"Erreur RAG: {e}")
    RAG_SYSTEM_OK = False

try:
    from llm_vanilla_translator import LLMVanillaTranslator
    VANILLA_SYSTEM_OK = True
except ImportError as e:
    st.error(f"Erreur Vanilla LLM: {e}")
    VANILLA_SYSTEM_OK = False

# Styles CSS améliorés
st.markdown("""
<style>
    .security-safe { background: #d4edda; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745; }
    .security-warning { background: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; }
    .security-danger { background: #f8d7da; padding: 10px; border-radius: 5px; border-left: 4px solid #dc3545; }
    .security-blocked { background: #f5f5f5; padding: 10px; border-radius: 5px; border-left: 4px solid #6c757d; }
    .hallucination-box { background: #ffe6e6; padding: 15px; border-radius: 8px; border: 2px solid #ff4444; margin: 10px 0; }
    .good-translation { background: #e6ffe6; padding: 15px; border-radius: 8px; border: 2px solid #44ff44; margin: 10px 0; }
    .comparison-header { 
        text-align: center; 
        padding: 10px; 
        margin: 5px 0; 
        border-radius: 5px; 
        font-weight: bold;
    }
    .vanilla-header { background: #fff3cd; color: #856404; }
    .rag-header { background: #d4edda; color: #155724; }
</style>
""", unsafe_allow_html=True)

# Initialisation des traducteurs
@st.cache_resource
def load_translators():
    """Charge les deux traducteurs"""
    translators = {}
    
    if RAG_SYSTEM_OK:
        with st.spinner("Chargement du système RAG+Patterns..."):
            translators['rag'] = SecurejQueryToJSTranslator(
                security_level="MEDIUM", 
                use_semgrep=True
            )
    
    if VANILLA_SYSTEM_OK:
        with st.spinner("Chargement du LLM Vanilla (GPT-2)..."):
            translators['vanilla'] = LLMVanillaTranslator(model_name="gpt2-medium")
    
    return translators

def analyze_hallucination(translation: str, original: str) -> dict:
    """Analyse si la traduction est hallucinée"""
    hallucination_indicators = [
        "var div={}", "new HTML5Canvas", "JSCredential", "UIView()",
        "Unable to generate", "// Error", "window[$(",
        "new Item()", "__construct", "I'm wearing a skirt"
    ]
    
    quality_indicators = [
        "document.getElementById", "document.querySelector", 
        "addEventListener", "classList", "textContent"
    ]
    
    hallucination_count = sum(1 for indicator in hallucination_indicators 
                            if indicator in translation)
    quality_count = sum(1 for indicator in quality_indicators 
                       if indicator in translation)
    
    is_hallucination = hallucination_count > 0 or quality_count == 0
    
    return {
        "is_hallucination": is_hallucination,
        "hallucination_score": hallucination_count,
        "quality_score": quality_count,
        "assessment": "HALLUCINATION" if is_hallucination else "CORRECT"
    }

def show_security_status(security_result):
    """Affiche le statut de sécurité"""
    level = security_result.get('level', SecurityLevel.SAFE)
    score = security_result.get('score', 0)
    
    if level == SecurityLevel.SAFE:
        st.markdown(f'<div class="security-safe"> SÉCURISÉ - Score: {score}/100</div>', 
                   unsafe_allow_html=True)
    elif level == SecurityLevel.WARNING:
        st.markdown(f'<div class="security-warning"> ATTENTION - Score: {score}/100</div>', 
                   unsafe_allow_html=True)
    elif level == SecurityLevel.DANGEROUS:
        st.markdown(f'<div class="security-danger">DANGEREUX - Score: {score}/100</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="security-blocked">BLOQUÉ</div>', 
                   unsafe_allow_html=True)

def show_comparison_result(vanilla_result, rag_result, jquery_code):
    """Affiche la comparaison des deux traductions"""
    
    # Analyse des hallucinations
    vanilla_analysis = analyze_hallucination(
        vanilla_result.get('translated', ''), jquery_code
    )
    
    # Headers de comparaison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="comparison-header vanilla-header">LLM Pur (GPT-2 Medium)</div>', 
                   unsafe_allow_html=True)
        
        # Affichage du résultat vanilla
        if vanilla_analysis['is_hallucination']:
            st.markdown(f'<div class="hallucination-box">'
                       f'<strong>⚠️ HALLUCINATION DÉTECTÉE</strong><br>'
                       f'<code>{vanilla_result.get("translated", "Erreur")}</code><br>'
                       f'<small>Score hallucination: {vanilla_analysis["hallucination_score"]}/10</small>'
                       f'</div>', unsafe_allow_html=True)
        else:
            st.code(vanilla_result.get('translated', 'Erreur'), language='javascript')
        
        # Métriques vanilla
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.metric("Qualité", f"{vanilla_analysis['quality_score']}/5")
        with col_v2:
            status = "Halluciné" if vanilla_analysis['is_hallucination'] else "✅ Correct"
            st.metric("Status", status)
    
    with col2:
        st.markdown('<div class="comparison-header rag-header">📚 RAG + Patterns + Sécurité</div>', 
                   unsafe_allow_html=True)
        
        # Affichage du résultat RAG
        rag_translation = rag_result.get('translated', 'Erreur')
        if 'BLOCKED' in rag_translation:
            st.markdown(f'<div class="security-blocked">'
                       f'<strong>TRADUCTION BLOQUÉE</strong><br>'
                       f'<code>{rag_translation}</code><br>'
                       f'<small>Sécurité prioritaire</small>'
                       f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="good-translation">'
                       f'<strong>TRADUCTION SÉCURISÉE</strong><br>'
                       f'<code>{rag_translation}</code>'
                       f'</div>', unsafe_allow_html=True)
        
        # Sécurité RAG
        if 'security' in rag_result:
            show_security_status(rag_result['security'])
        
        # Métriques RAG
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            score = rag_result.get('security', {}).get('score', 0)
            st.metric("Sécurité", f"{score}/100")
        with col_r2:
            method = rag_result.get('method', 'unknown')
            st.metric("Méthode", method.replace('_', ' ').title())

# Interface principale
def main():
    st.title("Étude Comparative: LLM vs RAG+Patterns")
    st.markdown("**Démonstration des Hallucinations IA** | Projet RNCP - IA Générative & Web")

    # Vérification des systèmes
    if not (RAG_SYSTEM_OK or VANILLA_SYSTEM_OK):
        st.error("Aucun système disponible. Vérifiez vos imports.")
        st.stop()

    # Chargement des traducteurs
    translators = load_translators()

    # Initialiser session_state si nécessaire
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = ""

    # --- Section des exemples d'hallucinations ---
    st.subheader("Exemples de Hallucinations")

    hallucination_examples = {
        "Click simple": "$('#btn').click(function() { alert('test'); });",
        "Hide elements": "$('.items').hide();",
        "Add class": "$('#box').addClass('active');",
        "Change text": "$('#title').text('Hello');",
        "Complex each": "$('.items').each(function(i) { $(this).addClass('item-' + i); });",
    }

    cols = st.columns(len(hallucination_examples))
    for i, (name, code) in enumerate(hallucination_examples.items()):
        with cols[i]:
            if st.button(name, key=f"ex_{name}", use_container_width=True):
                st.session_state.selected_example = code

    st.subheader("Code jQuery à tester")
    jquery_code = st.text_area(
        "Entrez le code jQuery:",
        value=st.session_state.selected_example,
        height=100,
        key="jquery_input"
    )

    # Mettre à jour session_state quand l'utilisateur modifie le texte
    if jquery_code != st.session_state.selected_example:
        st.session_state.selected_example = jquery_code

    if st.button("Comparer les Traductions", type="primary", use_container_width=True):
        if jquery_code.strip():
            st.subheader("Résultats Comparatifs")

            vanilla_result, rag_result = {}, {}
            vanilla_time, rag_time = 0, 0
            
            if 'vanilla' in translators:
                with st.spinner(" Traduction LLM Vanilla..."):
                    vanilla_start = time.time()
                    vanilla_result = translators['vanilla'].translate(jquery_code.strip())
                    vanilla_time = time.time() - vanilla_start

            if 'rag' in translators:
                with st.spinner("Traduction RAG+Patterns..."):
                    rag_start = time.time()
                    rag_result = translators['rag'].translate(jquery_code.strip())
                    rag_time = time.time() - rag_start

            if vanilla_result and rag_result:
                show_comparison_result(vanilla_result, rag_result, jquery_code)

                st.subheader("⚡ Performance")
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                with perf_col1:
                    st.metric("Temps LLM Vanilla", f"{vanilla_time:.2f}s")
                with perf_col2:
                    st.metric("Temps RAG+Patterns", f"{rag_time:.2f}s")
                with perf_col3:
                    difference = abs(vanilla_time - rag_time)
                    st.metric("Différence", f"{difference:.2f}s")
        else:
            st.warning("⚠️ Entrez du code jQuery pour comparer")

if __name__ == "__main__":
    main()