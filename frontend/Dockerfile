# syntax=docker/dockerfile:1

# ── Stage 1: install & build ──────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# NEXT_PUBLIC_API_URL is baked in at build time.
# Pass it as a build arg: --build-arg NEXT_PUBLIC_API_URL=https://...
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

# ── Stage 2: minimal runtime image ───────────────────────────────────────────
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

# Copy the standalone output produced by next.config.ts output: "standalone"
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

# Cloud Run injects PORT; Next.js standalone server respects this env var.
ENV PORT=8080
EXPOSE 8080

CMD ["node", "server.js"]
