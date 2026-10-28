from __future__ import annotations

import json

import streamlit as st

from auth.user_store import User, UserStore
from config.settings import AppSettings
from observability import configure_logging, log_event
from providers.base import ProviderName
from providers.model_registry import ModelRegistry
from ui.chat_view import render_chat_view


st.set_page_config(page_title="ResearchOS", page_icon="🔎", layout="wide")
settings = AppSettings.from_env()
configure_logging(settings.data_dir)
store = UserStore(settings.users_db)
LOGIN_COOKIE = "researchos_session"


def _begin_user_session(user: User, *, remember: bool) -> None:
    token = store.create_login_session(user.id, remember=remember)
    st.session_state["researchos_user"] = {"id": user.id, "username": user.username}
    st.session_state["researchos_session_token"] = token
    st.session_state["researchos_fresh_chat_user_id"] = user.id
    st.session_state["researchos_cookie_action"] = ("set", token, remember)


def _read_login_cookie() -> str | None:
    try:
        token = st.context.cookies.get(LOGIN_COOKIE)
    except Exception:
        return None
    return token if isinstance(token, str) else None


def _apply_cookie_action() -> None:
    action = st.session_state.pop("researchos_cookie_action", None)
    if not action:
        return
    kind, token, remember = action
    try:
        secure = str(st.context.url).startswith("https://")
    except Exception:
        secure = False
    if kind == "set":
        cookie = f"{LOGIN_COOKIE}={token}; Path=/; SameSite=Lax"
        if remember:
            cookie += "; Max-Age=2592000"
        if secure:
            cookie += "; Secure"
    else:
        cookie = f"{LOGIN_COOKIE}=; Path=/; SameSite=Lax; Max-Age=0"
        if secure:
            cookie += "; Secure"
    st.html(f"<script>document.cookie = {json.dumps(cookie)};</script>", unsafe_allow_javascript=True)


def authentication_view() -> None:
    st.title("ResearchOS")
    st.caption("A private research workspace for your local network.")
    sign_in, register = st.tabs(["Sign in", "Create account"])
    with sign_in:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            remember = st.checkbox("Remember me on this browser", value=True)
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            user = store.authenticate(username, password)
            if user is None:
                log_event("authentication_failed", status="rejected")
                st.error("Username or password was not recognized.")
            else:
                log_event("authentication_succeeded", user_id=user.id, status="authenticated")
                _begin_user_session(user, remember=remember)
                st.rerun()
    with register:
        st.info("New accounts are available to users who can reach this LAN service.")
        with st.form("register_form"):
            username = st.text_input("Choose a username", key="register_username")
            password = st.text_input("Choose a password", type="password", key="register_password")
            confirm = st.text_input("Confirm password", type="password", key="register_confirm")
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            if password != confirm:
                st.error("The passwords do not match.")
            else:
                try:
                    user = store.register(username, password)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    log_event("account_created", user_id=user.id, status="created")
                    _begin_user_session(user, remember=True)
                    st.success("Account created.")
                    st.rerun()


def provider_settings(user: User) -> None:
    st.subheader("Choose a generation model")
    st.caption("Select a provider and choose whether to use the common server key or your own private key.")
    saved = store.get_settings(user.id)
    providers = list(ProviderName)
    default_provider = saved.get("provider", ProviderName.GROQ.value)
    provider_labels = [provider.value for provider in providers]
    selected_label = st.selectbox(
        "Generation provider",
        provider_labels,
        index=provider_labels.index(default_provider) if default_provider in provider_labels else 0,
        key=f"provider_{user.id}",
    )
    provider = ProviderName(selected_label)
    if saved.get("provider") != provider.value:
        store.save_settings(user.id, {"provider": provider.value})
        saved = store.get_settings(user.id)
    elif saved.get("model") and not ModelRegistry.model_matches_provider(provider, str(saved["model"])):
        store.save_settings(user.id, {"provider": provider.value, "model": ""})
        saved = store.get_settings(user.id)
        st.warning("The saved model belongs to a different provider. Fetch this provider's model list and choose one to continue.")
    api_key_widget = f"personal_api_key_{user.id}_{provider.value}"
    has_personal_key = store.has_api_key(user.id, provider.value)
    shared_key = settings.groq_api_key if provider is ProviderName.GROQ else settings.gemini_api_key
    key_modes = saved.get("key_modes", {})
    saved_mode = key_modes.get(provider.value) if isinstance(key_modes, dict) else None
    default_mode = saved_mode or ("personal" if has_personal_key else ("shared" if shared_key else "personal"))
    source_labels = ["Use the common key", "Provide my own key"]
    source_modes = {source_labels[0]: "shared", source_labels[1]: "personal"}
    selected_source = st.radio(
        "Which API key should this account use?",
        source_labels,
        index=0 if default_mode == "shared" else 1,
        horizontal=True,
        key=f"api_key_source_{user.id}_{provider.value}",
    )
    key_mode = source_modes[selected_source]
    if saved_mode != key_mode:
        key_modes = {**(key_modes if isinstance(key_modes, dict) else {}), provider.value: key_mode}
        store.save_settings(user.id, {"provider": provider.value, "key_modes": key_modes})
        st.session_state.get("researchos_model_cache", {}).pop(f"{user.id}:{provider.value}", None)

    personal_key = None
    if key_mode == "shared":
        if shared_key:
            st.caption("This account will use the common key configured by the server administrator.")
        else:
            st.warning("The common key is not configured for this provider. Choose “Provide my own key” to continue.")
    else:
        if has_personal_key:
            st.caption(f"Your {provider.value} key is saved privately for this account and is used for its requests.")
            if st.button("Remove my API key", key=f"remove_api_key_{user.id}_{provider.value}"):
                store.delete_api_key(user.id, provider.value)
                st.session_state.get("researchos_model_cache", {}).pop(f"{user.id}:{provider.value}", None)
                st.rerun()
        personal_key = store.get_api_key(user.id, provider.value) if has_personal_key else None
        with st.form(f"personal_api_key_form_{user.id}_{provider.value}", clear_on_submit=True):
            entered_key = st.text_input(
                f"Your {provider.value} API key",
                type="password",
                key=api_key_widget,
                placeholder="Paste your key to use your own provider account",
            )
            save_api_key = st.form_submit_button("Save my API key")
        if save_api_key:
            try:
                store.save_api_key(user.id, provider.value, entered_key)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.get("researchos_model_cache", {}).pop(f"{user.id}:{provider.value}", None)
                st.success("Your API key was saved securely for this account.")
                st.rerun()

    registry = ModelRegistry.for_account(settings=settings, store=store, user_id=user.id)
    if not registry.configured(provider):
        st.warning("No key is available for the selected choice. Choose the common key or save your own key above.")
        return

    cache = st.session_state.setdefault("researchos_model_cache", {})
    cache_key = f"{user.id}:{provider.value}"
    cached = cache_key in cache
    fetch_col, clear_col, spacer = st.columns([1.5, 1.2, 3])
    if fetch_col.button("Fetch available models" if not cached else "Fetch models again",
                        type="primary", key=f"fetch_{user.id}_{provider.value}"):
        cache.pop(cache_key, None)
        st.session_state[f"model_fetch_requested_{user.id}_{provider.value}"] = True
    if cached and clear_col.button("Clear model list", key=f"clear_models_{user.id}_{provider.value}"):
        cache.pop(cache_key, None)
        st.session_state.pop(f"model_fetch_requested_{user.id}_{provider.value}", None)
        st.rerun()
    fetch_requested_key = f"model_fetch_requested_{user.id}_{provider.value}"
    if st.session_state.pop(fetch_requested_key, False):
        with st.spinner(f"Fetching {provider.value} text generation models…"):
            try:
                cache[cache_key] = ("ok", registry.get(provider).list_models())
                log_event("provider_models_loaded", user_id=user.id, provider=provider.value,
                          status="complete", result_count=len(cache[cache_key][1]))
            except Exception as exc:
                log_event("provider_model_list_failed", level=40, user_id=user.id,
                          provider=provider.value, status="failed", error_type=type(exc).__name__)
                message = str(exc).replace(settings.groq_api_key or "\0", "[redacted]")
                message = message.replace(settings.gemini_api_key or "\0", "[redacted]")
                message = message.replace(personal_key or "\0", "[redacted]")
                cache[cache_key] = ("error", f"Could not list models: {message[:300]}")
    if cache_key not in cache:
        st.info("Fetch the provider's model list to choose a model.")
        return
    status, value = cache[cache_key]
    if status == "error":
        st.error(value)
        st.caption("Check the selected API key and outbound network access to the provider API, then try fetching again.")
        return
    models = value
    if not models:
        st.info(f"{provider.value} returned no models compatible with text generation.")
        return
    model_ids = [model.id for model in models]
    prior_model = saved.get("model") if saved.get("provider") == provider.value else None
    if prior_model and not ModelRegistry.model_matches_provider(provider, str(prior_model)):
        prior_model = None
    selected_model = st.selectbox(
        "Model",
        model_ids,
        index=model_ids.index(prior_model) if prior_model in model_ids else 0,
        format_func=lambda model_id: next((m.display_name for m in models if m.id == model_id), model_id),
        key=f"model_{user.id}_{provider.value}",
    )
    store.save_settings(user.id, {"provider": provider.value, "model": selected_model})
    if st.button("Test selected model", key=f"test_{user.id}_{provider.value}"):
        with st.spinner("Testing model connection…"):
            result = registry.get(provider).test_connection(selected_model)
        if result.ok:
            log_event("provider_connection_test", user_id=user.id, provider=provider.value,
                      model=selected_model, status="complete")
            st.success(result.message)
        else:
            log_event("provider_connection_test", level=40, user_id=user.id, provider=provider.value,
                      model=selected_model, status="failed", error_type="ProviderConnectionError")
            st.error(result.message)


def main() -> None:
    # Keep the page heading clear of Streamlit's top chrome. The chat view also
    # renders below this content, so apply the spacing before any page is drawn.
    st.markdown(
        "<style>"
        "[data-testid='stMainBlockContainer']{padding-top:2.5rem!important;}"
        "[data-testid='stMainBlockContainer'] h1{line-height:1.25;overflow:visible;}"
        "</style>",
        unsafe_allow_html=True,
    )
    user_data = st.session_state.get("researchos_user")
    token = st.session_state.get("researchos_session_token")
    if isinstance(user_data, dict) and user_data.get("id") and isinstance(token, str):
        verified_user = store.authenticate_login_session(token)
        if verified_user is None or verified_user.id != user_data["id"]:
            log_event("login_session_invalid", status="rejected")
            st.session_state.pop("researchos_user", None)
            st.session_state.pop("researchos_session_token", None)
            st.session_state["researchos_cookie_action"] = ("clear", "", False)
        else:
            st.session_state["researchos_user"] = {
                "id": verified_user.id, "username": verified_user.username
            }
    else:
        if isinstance(user_data, dict) and user_data.get("id"):
            st.session_state.pop("researchos_user", None)
            st.session_state.pop("researchos_session_token", None)
        cookie_token = _read_login_cookie()
        restored_user = store.authenticate_login_session(cookie_token or "")
        if restored_user is not None:
            log_event("login_session_restored", user_id=restored_user.id, status="restored")
            st.session_state["researchos_user"] = {
                "id": restored_user.id, "username": restored_user.username
            }
            st.session_state["researchos_session_token"] = cookie_token
            st.session_state["researchos_fresh_chat_user_id"] = restored_user.id
        elif cookie_token:
            log_event("login_session_cookie_rejected", status="rejected")
            st.session_state["researchos_cookie_action"] = ("clear", "", False)
    _apply_cookie_action()
    user_data = st.session_state.get("researchos_user")
    if not isinstance(user_data, dict) or not user_data.get("id"):
        authentication_view()
        return
    user = User(user_data["id"], user_data["username"], "")
    account_dir = settings.user_data_dir(user.id)
    account_dir.mkdir(parents=True, exist_ok=True)
    st.title("ResearchOS")
    st.caption(f"Signed in as **{user.username}** · your conversations and knowledge stay in your account.")
    provider_settings(user)
    render_chat_view(user, settings, store)


if __name__ == "__main__":
    main()
