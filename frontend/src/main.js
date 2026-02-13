import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'

const routes = [
  { path: '/', component: () => import('./views/ReadingPane.vue') },
  { path: '/topics', component: () => import('./views/LifeTopicNavigator.vue') },
  { path: '/plans', component: () => import('./views/ReadingPlans.vue') },
  { path: '/study/:book/:chapter/:verse?', component: () => import('./views/StudyView.vue') },
  { path: '/discover', component: () => import('./views/ConceptExplorer.vue') },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

const pinia = createPinia()
const app = createApp(App)
app.use(pinia)
app.use(router)
app.mount('#app')
