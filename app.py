import os
from App import create_app
from dotenv import load_dotenv
load_dotenv()

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)

