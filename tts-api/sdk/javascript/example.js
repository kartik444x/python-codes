/**
 * sdk/javascript/example.js
 *
 * Node.js usage example for the TTS client SDK.
 */
const { TTSClient } = require('./tts_client');

async function main() {
    const client = new TTSClient({
        baseUrl: 'http://localhost:8000',
        apiKey: 'test-key-123',
    });

    console.log('Checking health...');
    const health = await client.checkHealth();
    console.log('Health:', health);

    console.log('\nSynthesizing speech to output.wav...');
    await client.synthesizeToFile({
        text: 'Hello from Node.js and JavaScript! The production TTS API works seamlessly.',
        voice: 'af_sarah',
        language: 'en',
        outputPath: './js_output.wav',
    });
    console.log('✓ Audio saved to ./js_output.wav');
}

main().catch(console.error);
