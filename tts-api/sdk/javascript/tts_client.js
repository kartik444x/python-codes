/**
 * sdk/javascript/tts_client.js
 *
 * Lightweight JavaScript / Node.js client for the TTS AI API.
 */
const fs = require('fs');
const path = require('path');

class TTSClient {
    /**
     * @param {Object} config
     * @param {string} [config.baseUrl='http://localhost:8000']
     * @param {string} [config.apiKey=null]
     */
    constructor({ baseUrl = 'http://localhost:8000', apiKey = null } = {}) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.apiKey = apiKey;
    }

    _getHeaders(extra = {}) {
        const headers = {
            'Content-Type': 'application/json',
            ...extra,
        };
        if (this.apiKey) {
            headers['X-API-Key'] = this.apiKey;
        }
        return headers;
    }

    /**
     * Check API health status.
     */
    async checkHealth() {
        const res = await fetch(`${this.baseUrl}/api/v1/health`, {
            headers: this._getHeaders(),
        });
        if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
        return await res.json();
    }

    /**
     * List available voices.
     */
    async listVoices({ language = null, gender = null } = {}) {
        const url = new URL(`${this.baseUrl}/api/v1/voices`);
        if (language) url.searchParams.append('language', language);
        if (gender) url.searchParams.append('gender', gender);

        const res = await fetch(url.toString(), {
            headers: this._getHeaders(),
        });
        if (!res.ok) throw new Error(`List voices failed: ${res.statusText}`);
        const data = await res.json();
        return data.voices || [];
    }

    /**
     * Synthesize speech synchronously and return an ArrayBuffer of audio bytes.
     */
    async synthesize({ text, voice = 'af_sarah', language = 'en', speed = 1.0, format = 'wav' }) {
        const res = await fetch(`${this.baseUrl}/api/v1/tts`, {
            method: 'POST',
            headers: this._getHeaders({ Accept: `audio/${format}` }),
            body: JSON.stringify({
                text,
                voice,
                language,
                speed,
                format,
                return_json: false,
            }),
        });
        if (!res.ok) {
            const err = await res.text();
            throw new Error(`TTS failed (${res.status}): ${err}`);
        }
        return await res.arrayBuffer();
    }

    /**
     * Synthesize speech and save to local disk file (Node.js).
     */
    async synthesizeToFile({ text, outputPath, voice = 'af_sarah', language = 'en', speed = 1.0, format = 'wav' }) {
        const buffer = await this.synthesize({ text, voice, language, speed, format });
        const dir = path.dirname(outputPath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        fs.writeFileSync(outputPath, Buffer.from(buffer));
        return outputPath;
    }

    /**
     * Submit an asynchronous TTS job for long-form content.
     */
    async synthesizeAsync({ text, voice = 'af_sarah', language = 'en', speed = 1.0, format = 'wav', webhookUrl = null }) {
        const res = await fetch(`${this.baseUrl}/api/v1/tts/async`, {
            method: 'POST',
            headers: this._getHeaders(),
            body: JSON.stringify({
                text,
                voice,
                language,
                speed,
                format,
                webhook_url: webhookUrl,
            }),
        });
        if (!res.ok) {
            const err = await res.text();
            throw new Error(`Async TTS submission failed: ${err}`);
        }
        const data = await res.json();
        return data.job_id;
    }

    /**
     * Poll an async job until completion.
     */
    async pollJob(jobId, { intervalMs = 1000, timeoutMs = 120000 } = {}) {
        const startTime = Date.now();
        while (Date.now() - startTime < timeoutMs) {
            const res = await fetch(`${this.baseUrl}/api/v1/jobs/${jobId}`, {
                headers: this._getHeaders(),
            });
            if (!res.ok) throw new Error(`Job poll failed: ${res.statusText}`);
            const data = await res.json();
            if (data.status === 'completed') {
                return data;
            } else if (data.status === 'failed' || data.status === 'cancelled') {
                throw new Error(`Job ${jobId} failed: ${data.error_message || data.status}`);
            }
            await new Promise((r) => setTimeout(r, intervalMs));
        }
        throw new Error(`Job ${jobId} timed out after ${timeoutMs}ms.`);
    }
}

module.exports = { TTSClient };
