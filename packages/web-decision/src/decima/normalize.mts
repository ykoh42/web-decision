const TASHKEEL = /[ؐ-ًؚ-ٰٟۖ-ۭ]/u;
const ARABIC_DIGITS = new Map([...'٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹'].map((value, index) => [value, String(index % 10)]));
const PERSIAN = new Map(Object.entries({ي: 'ی', ى: 'ی', ك: 'ک', ة: 'ه', ؤ: 'و', إ: 'ا', أ: 'ا'}));
const ARABIC = new Map(Object.entries({ی: 'ي', ک: 'ك'}));

/** Match Decima's NFC and Arabic-script normalization before tokenization. */
export function normalizeDecimaText(text: string, language = 'en'): string {
  let value = text.normalize('NFC');
  if (language !== 'fa' && language !== 'ar') return value.trim();
  value = [...value]
    .filter((character) => !TASHKEEL.test(character) && character !== 'ـ')
    .map((character) => ARABIC_DIGITS.get(character) ?? character)
    .map((character) => (language === 'fa' ? PERSIAN : ARABIC).get(character) ?? character)
    .join('');
  return value.replace(/[ \t]+/gu, ' ').trim();
}
