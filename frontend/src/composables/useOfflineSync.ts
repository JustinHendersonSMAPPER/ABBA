import { ref } from 'vue'
import type { Ref } from 'vue'
import { useApi } from './useApi'

const DB_NAME = 'abba-offline'
const DB_VERSION = 1
const STORE_NAME = 'verses'

interface SyncStatus {
  bookId: string
  chapter: number
  syncedAt: string
}

interface OfflineSyncReturn {
  syncing: Ref<boolean>
  syncProgress: Ref<number>
  syncedChapters: Ref<SyncStatus[]>
  syncChapter: (bookId: string, chapter: number) => Promise<void>
  getCachedData: (bookId: string, chapter: number) => Promise<Record<string, unknown> | null>
  getSyncStatus: () => Promise<SyncStatus[]>
  clearCache: () => Promise<void>
  isOnline: Ref<boolean>
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)

    request.onupgradeneeded = () => {
      const db = request.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' })
        store.createIndex('book_chapter', ['bookId', 'chapter'], { unique: true })
      }
    }

    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

function dbPut(db: IDBDatabase, data: Record<string, unknown>): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    store.put(data)
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

function dbGet(db: IDBDatabase, key: string): Promise<Record<string, unknown> | null> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const store = tx.objectStore(STORE_NAME)
    const request = store.get(key)
    request.onsuccess = () => resolve((request.result as Record<string, unknown>) || null)
    request.onerror = () => reject(request.error)
  })
}

function dbGetAll(db: IDBDatabase): Promise<Record<string, unknown>[]> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const store = tx.objectStore(STORE_NAME)
    const request = store.getAll()
    request.onsuccess = () => resolve(request.result as Record<string, unknown>[])
    request.onerror = () => reject(request.error)
  })
}

function dbClear(db: IDBDatabase): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    store.clear()
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

export function useOfflineSync(): OfflineSyncReturn {
  const api = useApi()
  const syncing = ref(false)
  const syncProgress = ref(0)
  const syncedChapters = ref<SyncStatus[]>([])
  const isOnline = ref(typeof navigator !== 'undefined' ? navigator.onLine : true)

  if (typeof window !== 'undefined') {
    window.addEventListener('online', () => { isOnline.value = true })
    window.addEventListener('offline', () => { isOnline.value = false })
  }

  async function syncChapter(bookId: string, chapter: number): Promise<void> {
    syncing.value = true
    syncProgress.value = 0
    try {
      const data = await api.mobileSync({
        book_ids: [bookId],
        chapters: [{ book_id: bookId, chapter }],
      })

      if (data) {
        const db = await openDB()
        const id = `${bookId}.${chapter}`
        await dbPut(db, {
          id,
          bookId,
          chapter,
          data,
          syncedAt: new Date().toISOString(),
        })
        db.close()

        syncProgress.value = 100
        await getSyncStatus()
      }
    } finally {
      syncing.value = false
    }
  }

  async function getCachedData(bookId: string, chapter: number): Promise<Record<string, unknown> | null> {
    try {
      const db = await openDB()
      const id = `${bookId}.${chapter}`
      const result = await dbGet(db, id)
      db.close()
      return result ? (result.data as Record<string, unknown>) : null
    } catch {
      return null
    }
  }

  async function getSyncStatus(): Promise<SyncStatus[]> {
    try {
      const db = await openDB()
      const all = await dbGetAll(db)
      db.close()
      const statuses = all.map(item => ({
        bookId: item.bookId as string,
        chapter: item.chapter as number,
        syncedAt: item.syncedAt as string,
      }))
      syncedChapters.value = statuses
      return statuses
    } catch {
      return []
    }
  }

  async function clearCache(): Promise<void> {
    try {
      const db = await openDB()
      await dbClear(db)
      db.close()
      syncedChapters.value = []
    } catch {
      // IndexedDB not available
    }
  }

  return {
    syncing,
    syncProgress,
    syncedChapters,
    syncChapter,
    getCachedData,
    getSyncStatus,
    clearCache,
    isOnline,
  }
}
