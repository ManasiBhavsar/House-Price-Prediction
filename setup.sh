mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"manasibhavsar77@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\n
" > /.streamlit/config.toml
