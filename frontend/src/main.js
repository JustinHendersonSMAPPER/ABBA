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
    path: '/search',
    name: 'search',
    component: () => import('./views/SearchResults.vue'),
    meta: { title: 'Search' },
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
    path: '/compare',
    name: 'compare',
    component: () => import('./views/TranslationCompare.vue'),
    meta: { title: 'Compare Translations' },
  },
  {
    path: '/lexicon/:strongs?',
    name: 'lexicon',
    component: () => import('./views/LexiconView.vue'),
    meta: { title: 'Word Study' },
  },
  {
    path: '/collections',
    name: 'collections',
    component: () => import('./views/CollectionsView.vue'),
    meta: { title: 'My Collections' },
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
