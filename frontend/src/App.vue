<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import {
  ArrowUp, BookOpen, Check, ChevronRight, CircleAlert, FileText,
  Layers, LoaderCircle, MessageSquarePlus, Plus, RefreshCw, Trash2, X,
} from 'lucide-vue-next'
import { request } from './api'
import type { Answer, Citation, Document, Health } from './api'

interface Turn { id: number; question: string; result?: Answer; error?: string }
const documents = ref<Document[]>([])
const health = ref<Health | null>(null)
const healthError = ref('')
const error = ref('')
const uploadStatus = ref('')
const question = ref('')
const turns = ref<Turn[]>([])
const uploading = ref(false)
const asking = ref(false)
const deleting = ref<string | null>(null)
const loading = ref(true)
const activeCitation = ref<Citation | null>(null)
const sourceDialog = ref<HTMLDialogElement | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)
const messages = ref<HTMLElement | null>(null)
const dragging = ref(false)
const busy = computed(() => uploading.value || asking.value || deleting.value !== null)
const modelReady = computed(() => health.value?.status === 'ok')
const modelStatus = computed(() => {
  if (healthError.value) return healthError.value
  if (!health.value) return 'Checking models'
  if (health.value.ollama === 'unavailable') return 'Ollama unavailable'
  if (health.value.missing_models.length) return `Missing: ${health.value.missing_models.join(', ')}`
  return 'Models ready'
})

function messageOf(caught: unknown) {
  return caught instanceof Error ? caught.message : 'Something went wrong. Please try again.'
}

async function checkHealth() {
  try {
    health.value = await request<Health>('/health')
    healthError.value = ''
  } catch (caught) { healthError.value = messageOf(caught) }
}

async function loadDocuments() {
  try { documents.value = await request<Document[]>('/documents') }
  catch (caught) { error.value = messageOf(caught) }
  finally { loading.value = false }
}

async function upload(file?: File) {
  if (!file || busy.value) return
  error.value = ''
  uploadStatus.value = ''
  if (!/\.(pdf|txt)$/i.test(file.name)) {
    error.value = 'Choose a PDF or UTF-8 TXT document.'
    return
  }
  uploading.value = true
  uploadStatus.value = `Indexing ${file.name}`
  const body = new FormData()
  body.append('file', file)
  try {
    const document = await request<Document>('/documents', { method: 'POST', body })
    documents.value.unshift(document)
    uploadStatus.value = `${document.filename} is ready`
  } catch (caught) {
    error.value = messageOf(caught)
    uploadStatus.value = ''
  } finally {
    uploading.value = false
    if (fileInput.value) fileInput.value.value = ''
    await checkHealth()
  }
}

function selectFile(event: Event) {
  void upload((event.target as HTMLInputElement).files?.[0])
}

function dropFile(event: DragEvent) {
  dragging.value = false
  if ((event.dataTransfer?.files.length ?? 0) > 1) {
    error.value = 'Upload one document at a time.'
    return
  }
  void upload(event.dataTransfer?.files[0])
}

async function removeDocument(document: Document) {
  if (busy.value || !window.confirm(`Delete ${document.filename} and its indexed passages?`)) return
  deleting.value = document.id
  error.value = ''
  try {
    await request<void>(`/documents/${document.id}`, { method: 'DELETE' })
    documents.value = documents.value.filter(item => item.id !== document.id)
    turns.value = []
    activeCitation.value = null
    uploadStatus.value = ''
  } catch (caught) { error.value = messageOf(caught) }
  finally { deleting.value = null }
}

async function scrollToLatest() {
  await nextTick()
  messages.value?.scrollTo({ top: messages.value.scrollHeight, behavior: 'smooth' })
}

async function ask() {
  const text = question.value.trim()
  if (!text || busy.value || !documents.value.length) return
  const turn: Turn = { id: Date.now(), question: text }
  turns.value.push(turn)
  question.value = ''
  asking.value = true
  activeCitation.value = null
  await scrollToLatest()
  try {
    const result = await request<Answer>('/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text }),
    })
    const current = turns.value.find(item => item.id === turn.id)
    if (current) current.result = result
  } catch (caught) {
    const current = turns.value.find(item => item.id === turn.id)
    if (current) current.error = messageOf(caught)
    await checkHealth()
  } finally {
    asking.value = false
    await scrollToLatest()
  }
}

function newConversation() {
  if (busy.value) return
  turns.value = []
  activeCitation.value = null
}

watch(activeCitation, async citation => {
  await nextTick()
  if (citation) sourceDialog.value?.showModal()
  else sourceDialog.value?.close()
})

onMounted(() => { void loadDocuments(); void checkHealth() })
</script>

<template>
  <div class="workspace">
    <header class="topbar">
      <a class="brand" href="/" aria-label="Local AI Knowledge Assistant home">
        <span class="brand-mark"><Layers :size="23" /></span>
        <span>Local AI <strong>Knowledge Assistant</strong></span>
      </a>
      <span class="local-label"><span class="status-dot" /> LOCAL WORKSPACE</span>
    </header>

    <aside class="library">
      <div class="section-title"><h1>Documents</h1><span class="count">{{ documents.length }}</span></div>
      <button class="upload-button" :class="{ dragging }" :disabled="busy"
        @click="fileInput?.click()" @dragover.prevent="dragging = true"
        @dragleave.prevent="dragging = false" @drop.prevent="dropFile">
        <LoaderCircle v-if="uploading" :size="18" class="spin" /><Plus v-else :size="18" />
        {{ uploading ? 'Indexing document' : 'Add document' }}
        <span class="upload-format">PDF / TXT</span>
      </button>
      <input ref="fileInput" class="visually-hidden" type="file" accept=".pdf,.txt" @change="selectFile" />
      <div v-if="uploadStatus" class="ingestion-status" role="status">
        <LoaderCircle v-if="uploading" :size="14" class="spin" /><Check v-else :size="14" />
        <span>{{ uploadStatus }}</span>
      </div>
      <div v-if="error" class="error-banner" role="alert"><CircleAlert :size="16" /><span>{{ error }}</span></div>
      <div class="document-list">
        <p v-if="loading" class="muted">Loading documents...</p>
        <div v-else-if="!documents.length" class="library-empty"><FileText :size="28" /><p>No documents yet</p></div>
        <div v-for="document in documents" :key="document.id" class="document-row">
          <FileText :size="20" class="file-icon" />
          <div class="document-info"><span class="filename">{{ document.filename }}</span>
            <span class="document-meta">{{ document.chunks }} passages <span class="ready-dot" /> Ready</span>
          </div>
          <button class="icon-button delete-button" :disabled="busy" :title="`Delete ${document.filename}`"
            :aria-label="`Delete ${document.filename}`" @click="removeDocument(document)">
            <LoaderCircle v-if="deleting === document.id" :size="16" class="spin" /><Trash2 v-else :size="16" />
          </button>
        </div>
      </div>
      <div class="model-panel">
        <div class="model-heading"><span>MODEL STATUS</span><button class="icon-button" title="Refresh model status"
          aria-label="Refresh model status" @click="checkHealth"><RefreshCw :size="14" /></button></div>
        <p class="model-status" :class="{ degraded: !modelReady }"><span class="status-dot" />{{ modelStatus }}</p>
        <span v-if="health" class="model-name">{{ health.chat_model }}</span>
      </div>
    </aside>

    <main class="chat">
      <div class="chat-toolbar"><div><span class="eyebrow">WORKSPACE</span><h2>Conversation</h2></div>
        <button class="icon-button" title="New conversation" aria-label="New conversation" :disabled="busy || !turns.length"
          @click="newConversation"><MessageSquarePlus :size="20" /></button>
      </div>
      <div ref="messages" class="messages" aria-live="polite" aria-relevant="additions text">
        <div v-if="!turns.length" class="welcome">
          <div class="welcome-symbol"><BookOpen :size="35" :stroke-width="1.3" /></div>
          <span class="eyebrow">YOUR KNOWLEDGE, CONNECTED</span>
          <h2>A little more clarity.</h2>
          <p>{{ documents.length ? `${documents.length} document${documents.length === 1 ? '' : 's'} ready` : 'No sources in this workspace' }}</p>
        </div>
        <article v-for="turn in turns" :key="turn.id" class="turn">
          <div class="question"><span class="speaker">YOU</span><p>{{ turn.question }}</p></div>
          <div class="response"><span class="speaker assistant-speaker"><Layers :size="14" /> ASSISTANT</span>
            <template v-if="turn.result">
              <p class="answer-text" :class="{ abstention: turn.result.insufficient_evidence }">{{ turn.result.answer }}</p>
              <div v-if="turn.result.citations.length" class="citations">
                <button v-for="citation in turn.result.citations" :key="citation.label" class="citation-button"
                  :aria-label="`Source ${citation.label}: ${citation.filename}`" @click="activeCitation = citation">
                  <span class="citation-number">{{ citation.label }}</span><FileText :size="14" />
                  <span>{{ citation.filename }} <small>{{ citation.page ? `p. ${citation.page}` : `section ${citation.section}` }}</small></span>
                  <ChevronRight :size="14" />
                </button>
              </div>
            </template>
            <p v-else-if="turn.error" class="turn-error" role="alert">{{ turn.error }}</p>
            <div v-else class="thinking" role="status"><LoaderCircle :size="16" class="spin" /> Searching evidence...</div>
          </div>
        </article>
      </div>
      <form class="composer" @submit.prevent="ask">
        <label class="visually-hidden" for="question">Your question</label>
        <textarea id="question" v-model="question" rows="2" maxlength="2000" :disabled="busy || !documents.length"
          :placeholder="documents.length ? 'Ask a question...' : 'Add a document to begin'"
          @keydown.enter.exact.prevent="ask" />
        <div class="composer-bottom"><span>{{ documents.length }} {{ documents.length === 1 ? 'source' : 'sources' }}</span>
          <button class="send-button" type="submit" title="Send question" aria-label="Send question"
            :disabled="busy || !question.trim() || !documents.length"><LoaderCircle v-if="asking" :size="19" class="spin" /><ArrowUp v-else :size="20" /></button>
        </div>
      </form>
      <footer class="chat-footer">Local inference <span>/</span> {{ health?.embedding_model || 'Ollama embeddings' }}</footer>
    </main>

    <dialog ref="sourceDialog" class="source-overlay" aria-labelledby="source-title"
      @click.self="activeCitation = null" @cancel.prevent="activeCitation = null"
      @keydown.esc.prevent.stop="activeCitation = null">
      <section v-if="activeCitation" class="source-panel">
        <div class="source-header"><span class="eyebrow">SOURCE {{ activeCitation.label }}</span>
          <button class="icon-button" title="Close source" aria-label="Close source" autofocus @click="activeCitation = null"><X :size="20" /></button></div>
        <FileText :size="28" class="source-file-icon" />
        <h2 id="source-title">{{ activeCitation.filename }}</h2>
        <p class="source-location">{{ activeCitation.page ? `Page ${activeCitation.page}` : `Text section ${activeCitation.section}` }}</p>
        <blockquote>{{ activeCitation.text }}</blockquote>
      </section>
    </dialog>
  </div>
</template>