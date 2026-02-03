import os
import random


def build_kuai_proxy_url(username, password, host, port=None):
    if port:
        return f"http://{username}:{password}@{host}:{port}"
    return f"http://{username}:{password}@{host}"


def _fetch_kdl_proxy_list(secret_id, secret_key, num=1, sign_type="token", api="dps", **params):
    try:
        import kdl  # type: ignore
    except Exception as exc:
        raise RuntimeError("kdl SDK not installed, run: pip install kdl") from exc

    auth = kdl.Auth(secret_id, secret_key)
    client = kdl.Client(auth)
    if api == "kps":
        return client.get_kps(num, sign_type=sign_type, **params)
    return client.get_dps(num, sign_type=sign_type, **params)


def enable_kuai_proxy(
    username,
    password,
    host="tps.kdlapi.com",
    port=15818,
    mode="tps",
    secret_id=None,
    secret_key=None,
    sign_type="token",
    api="dps",
    api_params=None,
):
    api_params = api_params or {}
    if mode == "api":
        if not secret_id or not secret_key:
            raise ValueError("KDL_SECRET_ID/KDL_SECRET_KEY required for api mode")
        proxy_list = _fetch_kdl_proxy_list(
            secret_id, secret_key, num=api_params.get("num", 1), sign_type=sign_type, api=api, **api_params
        )
        if not proxy_list:
            raise RuntimeError("empty proxy list from kdl api")
        proxy_host = random.choice(proxy_list)
        proxy_url = build_kuai_proxy_url(username, password, proxy_host, None)
    else:
        proxy_url = build_kuai_proxy_url(username, password, host, port)

    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["ALL_PROXY"] = proxy_url
    return proxy_url


def disable_proxy():
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
