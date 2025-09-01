#!/usr/bin/env python3
"""
app.py - Interface Streamlit Simple
Copiez ce code exactement dans un fichier nommé 'app.py'
"""

import streamlit as st
import time

# Configuration de la page
st.set_page_config(
    page_title="jQuery → JS Translator",
    page_icon="🔄",
    layout="wide"
)

# Import de votre système
try:
    from translator import SecurejQueryToJSTranslator, SecurityLevel
    SYSTEM_OK = True
except ImportError as e:
    st.error(f"❌ Erreur: {e}")
    st.info("Vérifiez que translator.py et semgrep_validator.py sont dans le même dossier")
    SYSTEM_OK = False

# Styles CSS
st.markdown("""
<style>
    .security-safe { background: #d4edda; padding: 10px; border-radius: 5px; }
    .security-warning { background: #fff3cd; padding: 10px; border-radius: 5px; }
    .security-danger { background: #f8d7da; padding: 10px; border-radius: 5px; }
    .security-blocked { background: #f5f5f5; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Initialisation
if 'translator' not in st.session_state and SYSTEM_OK:
    with st.spinner("🚀 Chargement..."):
        st.session_state.translator = SecurejQueryToJSTranslator(
            security_level="MEDIUM", 
            use_semgrep=True
        )

def show_security_status(security_result):
    """Affiche le statut de sécurité"""
    level = security_result.get('level', SecurityLevel.SAFE)
    score = security_result.get('score', 0)
    
    if level == SecurityLevel.SAFE:
        st.markdown(f'<div class="security-safe">✅ SÉCURISÉ - Score: {score}/100</div>', 
                   unsafe_allow_html=True)
    elif level == SecurityLevel.WARNING:
        st.markdown(f'<div class="security-warning">⚠️ ATTENTION - Score: {score}/100</div>', 
                   unsafe_allow_html=True)
    elif level == SecurityLevel.DANGEROUS:
        st.markdown(f'<div class="security-danger">❌ DANGEREUX - Score: {score}/100</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="security-blocked">⛔ BLOQUÉ</div>', 
                   unsafe_allow_html=True)

# Interface principale
def main():
    st.title("🔄 jQuery → Vanilla JS Translator")
    st.markdown("**IA Générative & Sécurité Web** | Projet RNCP")
    
    if not SYSTEM_OK:
        st.stop()
    
    # Initialiser la valeur par défaut
    if 'current_code' not in st.session_state:
        st.session_state.current_code = ""
    
    # Exemples en haut
    st.subheader("📚 Exemples rapides")
    examples = {
        "Click event": "$('#btn').click(function() { alert('Hello'); });",
        "Hide element": "$('.box').hide();",
        "Add class": "$('#element').addClass('active');",
        "Change text": "$('#title').text('New title');",
        "Unsafe (bloqué)": "$('#content').html(userInput);"
    }
    
    # Boutons d'exemples
    cols = st.columns(len(examples))
    for i, (name, code) in enumerate(examples.items()):
        with cols[i]:
            if st.button(name, key=name, use_container_width=True):
                st.session_state.current_code = code
    
    # Colonnes principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Code jQuery")
        jquery_code = st.text_area(
            "Entrez ou modifiez le code:",
            value=st.session_state.current_code,
            height=250,
            key="jquery_input"
        )
        
        # Mettre à jour le state quand l'utilisateur tape
        if jquery_code != st.session_state.current_code:
            st.session_state.current_code = jquery_code
    
    # Bouton de traduction
    if st.button("🚀 Traduire", type="primary", use_container_width=True):
        if st.session_state.current_code.strip():
            with st.spinner("⚡ Traduction en cours..."):
                start_time = time.time()
                result = st.session_state.translator.translate(st.session_state.current_code.strip())
                translation_time = time.time() - start_time
            
            # Résultats
            with col2:
                st.subheader("⚡ Vanilla JavaScript")
                st.code(result['translated'], language='javascript')
            
            # Analyse de sécurité
            st.subheader("🛡️ Analyse de Sécurité")
            show_security_status(result['security'])
            
            # Métriques
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("Score", f"{result['security'].get('score', 0)}/100")
            with col4:
                safe = "✅ Oui" if result['security'].get('safe_to_execute') else "❌ Non"
                st.metric("Sûr", safe)
            with col5:
                st.metric("Temps", f"{translation_time:.2f}s")
            
            # Issues de sécurité
            issues = result['security'].get('issues', [])
            if issues:
                st.subheader("🚨 Problèmes détectés")
                for issue in issues:
                    st.write(f"• **{issue.get('type')}**: {issue.get('message')}")
            else:
                st.success("🔒 Aucun problème de sécurité détecté")
        else:
            st.warning("⚠️ Entrez du code jQuery d'abord")
    
    # Informations sur le projet
    st.divider()
    st.header("📋 Contexte de Thèse")
    
    col6, col7, col8 = st.columns(3)
    with col6:
        st.info("**🤖 IA Générative**\nTraduction automatique avec GPT-2")
    with col7:
        st.info("**📚 RAG System**\nPatterns sécurisés pour éviter les hallucinations")
    with col8:
        st.info("**🛡️ Sécurité**\nValidation Semgrep + analyse éthique")

if __name__ == "__main__":
    main()