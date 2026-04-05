# Fast Docker Builds with BuildKit Cache

This project uses Docker BuildKit for efficient layer caching. Without it, large dependencies like `zstandard` are re-downloaded on every build.

## Enable BuildKit (Required for Fast Builds)

**Option 1: Set environment variable (recommended)**
```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT_PROGRESS=plain

# Now all builds will be fast
docker compose build mitm-proxy
docker compose build validator
```

**Option 2: Enable permanently in daemon config**
Edit `~/.docker/daemon.json`:
```json
{
  "features": {
    "buildkit": true
  }
}
```

Then restart Docker:
```bash
systemctl restart docker  # or: brew services restart docker
```

## How It Works

1. **First build (cold cache)**: Downloads all dependencies (~2-3 min)
2. **Subsequent builds with code changes**: Only rebuilds source layer (~5-10 sec)
3. **Dependency-only changes**: Only rebuilds dependency layer (~30-60 sec, reuses pip/poetry/model caches)

### Key optimizations:

- ✅ `.dockerignore` prevents source code from invalidating dependency layer
- ✅ `--mount=type=cache,sharing=locked` persists pip/poetry cache across builds
- ✅ Host volume mounts (`~/.cache/pip`, `~/.cache/huggingface`, etc.) share downloads with host
- ✅ BuildKit syntax `# syntax=docker/dockerfile:1.4` enables advanced caching

## Verify BuildKit is Enabled

```bash
docker buildx ls
# Should show: buildx* | linux/amd64, linux/arm64 (with asterisk = active)
```

Or check build output:
```bash
DOCKER_BUILDKIT=1 docker compose build validator 2>&1 | grep -i buildkit
```

## If You're Still Seeing Slow Builds

1. **Verify BuildKit is enabled**:
   ```bash
   DOCKER_BUILDKIT=1 docker build --help | grep buildkit
   ```

2. **Check Docker daemon config**:
   ```bash
   docker version | grep -A2 Server
   # Should show BuildKit: true
   ```

3. **Clear cached images**:
   ```bash
   docker system prune -a --volumes
   docker compose build --no-cache validator  # Full rebuild with fresh layers
   ```

4. **Verify host caches are mounted**:
   ```bash
   docker compose run --rm validator ls -lh /root/.cache/pip
   # Should show existing wheels from your host machine
   ```

## Expected Build Times (with BuildKit + host caches)

| Scenario | Time |
|----------|------|
| Cold build (no cache) | 2-3 min |
| Code change only | 5-10 sec |
| Dependency change | 30-60 sec |
| Rebuild same code | 1-2 sec (fully cached) |
