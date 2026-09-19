FROM node:24-bookworm-slim AS build
WORKDIR /app/front
COPY front/package.json front/package-lock.json ./
RUN npm ci
COPY front/ ./
# Browser requests go to the same origin; no server IP or secret is compiled in.
ENV VITE_API_BASE_URL=/api VITE_TENANT_ID=default
RUN npm test && npm run build

FROM caddy:2.11.4-alpine
COPY infra/deploy/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /app/front/dist /srv
