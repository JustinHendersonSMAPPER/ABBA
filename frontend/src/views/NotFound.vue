<template>
  <div class="not-found">
    <h1 class="not-found-code">404</h1>
    <p class="not-found-message">Page not found</p>
    <p class="not-found-hint">
      The page you're looking for doesn't exist.
    </p>
    <router-link to="/" class="home-link">Back to Reading</router-link>
  </div>
</template>

<style scoped>
.not-found {
  text-align: center;
  padding: 4rem 1rem;
  font-family: var(--font-ui);
}

.not-found-code {
  font-size: 4rem;
  font-weight: 700;
  color: var(--color-accent);
  margin-bottom: 0.25rem;
}

.not-found-message {
  font-size: 1.25rem;
  margin-bottom: 0.5rem;
}

.not-found-hint {
  font-size: 0.9rem;
  opacity: 0.6;
  margin-bottom: 1.5rem;
}

.home-link {
  display: inline-block;
  padding: 0.5rem 1.25rem;
  background: var(--color-accent);
  color: white;
  text-decoration: none;
  border-radius: 6px;
  font-size: 0.9rem;
}

.home-link:hover {
  opacity: 0.9;
}
</style>
