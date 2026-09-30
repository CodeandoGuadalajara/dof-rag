"""Browser streaming over the existing authenticated, replayable SSE endpoint."""
CHAT_SCRIPT = r"""
(() => {
  const refresh = async (response, keepDraft = false) => {
    const composer = document.querySelector('#chat-question');
    const draft = composer.value;
    const focused = document.activeElement === composer;
    const previous = document.querySelector('[data-chat-messages]');
    const scroll = previous.scrollTop;
    const follow = scroll + previous.clientHeight >= previous.scrollHeight - 120;
    const html = await response.text();
    const page = new DOMParser().parseFromString(html, 'text/html');
    const replacement = page.querySelector('#chat-body');
    if (!replacement) { location.href = '/login?next=%2Fchat'; return; }
    document.querySelector('#chat-body').replaceWith(replacement);
    const next = document.querySelector('[data-chat-messages]');
    next.scrollTop = follow ? next.scrollHeight : scroll;
    const input = document.querySelector('#chat-question');
    if (keepDraft) input.value = draft;
    if (focused) input.focus({preventScroll: true});
    start();
  };
  const start = () => {
    const node = document.querySelector('[data-chat-events]');
    if (!node || !window.EventSource) return;
    const status = node.querySelector('[data-chat-status]');
    const activity = node.querySelector('[data-chat-activity]');
    const scroller = document.querySelector('[data-chat-messages]');
    const turns = new Map(), tools = new Map();
    const turnNode = (number) => {
      if (turns.has(number)) return turns.get(number);
      const section = document.createElement('section');
      const thought = document.createElement('details'); thought.open = true;
      const summary = document.createElement('summary'); summary.textContent = `Pensamiento del modelo · turno ${number}`;
      const reasoning = document.createElement('pre');
      thought.append(summary, reasoning);
      const answer = document.createElement('div'); answer.style.whiteSpace = 'pre-wrap';
      section.append(thought, answer); activity.append(section);
      const value = {reasoning, answer}; turns.set(number, value); return value;
    };
    const source = new EventSource(node.dataset.chatEvents);
    let last = 0;
    source.addEventListener('progress', (message) => {
      const event = JSON.parse(message.data);
      if (event.sequence <= last) return;
      last = event.sequence;
      const payload = event.payload || {};
      const follow = scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 120;
      if (payload.chat_delta) {
        const turn = turnNode(payload.turn);
        const target = payload.chat_delta === 'reasoning_content' ? turn.reasoning : turn.answer;
        target.append(document.createTextNode(payload.text || ''));
        status.textContent = 'Generando…';
      } else if (event.event_type === 'tool_started') {
        const details = document.createElement('details');
        const summary = document.createElement('summary'); summary.textContent = `Herramienta: ${payload.tool}`;
        const argumentsNode = document.createElement('pre'); argumentsNode.textContent = JSON.stringify(payload.arguments, null, 2);
        const result = document.createElement('pre'); result.textContent = 'Ejecutando…';
        details.append(summary, argumentsNode, result); activity.append(details);
        tools.set(payload.call_id, result);
        status.textContent = payload.message || 'Consultando el DOF…';
      } else if (event.event_type === 'tool_completed') {
        const result = tools.get(payload.call_id);
        if (result) result.textContent = JSON.stringify(payload.output, null, 2);
      } else if (payload.message) {
        status.textContent = payload.message;
      }
      if (follow) scroller.scrollTop = scroller.scrollHeight;
    });
    source.addEventListener('queue', (message) => { status.textContent = JSON.parse(message.data).message; });
    source.addEventListener('terminal', async () => {
      source.close();
      try { await refresh(await fetch('/chat', {credentials: 'same-origin', cache: 'no-store'}), true); }
      catch (_) { status.textContent = 'La consulta terminó. Recarga para ver el resultado guardado.'; }
    });
    source.onerror = () => { status.textContent = 'Reconectando… Tu consulta sigue ejecutándose.'; };
  };
  document.addEventListener('submit', async (event) => {
    const form = event.target;
    if (!form.matches('[data-chat-form]') || !window.EventSource) return;
    event.preventDefault();
    if (form.dataset.sending || form.querySelector('button').disabled) return;
    form.dataset.sending = 'true';
    const body = new FormData(form);
    const button = form.querySelector('button'); button.disabled = true;
    try {
      const response = await fetch('/chat', {method: 'POST', body, credentials: 'same-origin'});
      await refresh(response);
      const scroller = document.querySelector('[data-chat-messages]');
      scroller.scrollTop = scroller.scrollHeight;
    } catch (_) {
      delete form.dataset.sending;
      button.disabled = false;
      document.querySelector('[data-chat-error]').textContent = 'Error de conexión. Puedes reenviar: no se cobrará dos veces el mismo mensaje.';
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.target.id === 'chat-question' && event.key === 'Enter' && !event.shiftKey && !event.isComposing
        && !event.target.form.querySelector('button').disabled) {
      event.preventDefault(); event.target.form.requestSubmit();
    }
  });
  const scroller = document.querySelector('[data-chat-messages]');
  scroller.scrollTop = scroller.scrollHeight;
  start();
})();
"""
