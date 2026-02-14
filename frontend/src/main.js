import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'

const routes = [
  {
    path: '/',
    name: 'read',
    component: () => import('./views/ReadingPane.vue'),
    meta: { title: 'Read' },
  },
  {
    path: '/topics',
    name: 'topics',
    component: () => import('./views/LifeTopicNavigator.vue'),
    meta: { title: 'Life Topics' },
  },
  {
    path: '/plans',
    name: 'plans',
    component: () => import('./views/ReadingPlans.vue'),
    meta: { title: 'Reading Plans' },
  },
  {
    path: '/study/:book/:chapter/:verse?',
    name: 'study',
    component: () => import('./views/StudyView.vue'),
    meta: { title: 'Study' },
  },
  {
    path: '/discover',
    name: 'discover',
    component: () => import('./views/ConceptExplorer.vue'),
    meta: { title: 'Discover' },
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: () => import('./views/NotFound.vue'),
    meta: { title: 'Not Found' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(_to, _from, savedPosition) {
    return savedPosition || { top: 0 }
  },
})

router.afterEach((to) => {
  const base = 'ABBA Bible Study'
  document.title = to.meta.title ? `${to.meta.title} | ${base}` : base
})

const pinia = createPinia()
const app = createApp(App)
app.use(pinia)
app.use(router)
app.mount('#app')
