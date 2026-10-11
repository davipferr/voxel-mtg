/** One-line notices at the top of the screen. Errors stay until dismissed; info fades after 4 s. */
export function createToast(root: HTMLElement) {
  const msg = root.querySelector<HTMLElement>('.msg')!;
  let timer = 0;
  root.querySelector('button')!.addEventListener('click', () => (root.hidden = true));
  return (text: string, kind: 'info' | 'error' = 'info') => {
    clearTimeout(timer);
    msg.textContent = text;
    root.hidden = false;
    if (kind === 'info') timer = window.setTimeout(() => (root.hidden = true), 4000);
  };
}
