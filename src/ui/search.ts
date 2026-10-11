import { autocomplete } from '../data/scryfall';

/**
 * Search box with name suggestions from Scryfall's autocomplete, in a `<datalist>`. Suggestions are
 * fetched 200 ms after the last keystroke, and a stale answer never replaces a newer one.
 */
export function bindSearch(form: HTMLFormElement, input: HTMLInputElement, datalist: HTMLDataListElement, onGo: (name: string) => void): void {
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (input.value.trim()) onGo(input.value.trim());
    input.select();
  });

  let timer = 0;
  let latest = 0;
  input.addEventListener('input', (e) => {
    // Picking a suggestion fills the box with the option's full value, with no keypress; go straight away.
    if (!(e instanceof InputEvent) || e.inputType === 'insertReplacementText') {
      if ([...datalist.options].some((o) => o.value === input.value)) {
        onGo(input.value);
        return;
      }
    }
    clearTimeout(timer);
    const text = input.value;
    timer = window.setTimeout(() => {
      const mine = ++latest;
      autocomplete(text)
        .then((names) => {
          if (mine !== latest) return;
          datalist.replaceChildren(...names.map((n) => new Option(n, n)));
        })
        .catch(() => {});
    }, 200);
  });
}
