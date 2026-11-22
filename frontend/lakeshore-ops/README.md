# React + Vite

## API configuration

The frontend reads the backend URL from `VITE_API_BASE_URL`.

1. Copy `.env.example` to `.env.local`.
2. Set the value, for example `VITE_API_BASE_URL=http://localhost:8000`.
3. Restart Vite after changing the environment file.

Run the backend with `python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000`.

Backend environment variables:

- `API_HOST`: Uvicorn bind address (default `0.0.0.0`)
- `API_PORT`: Uvicorn port (default `8000`)
- `CORS_ORIGINS`: comma-separated frontend origins (default includes Vite's localhost origins)

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
