"""Synthetic invoice content with exact ground truth.

Everything is drawn from a seeded ``random.Random`` so a dataset is reproducible from its
seed. The ground truth is an ``Invoice`` whose every ``quote`` is the exact string the
renderer will print, which is what makes "did the model quote the source correctly"
measurable later.
"""

import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from doc_intel.models import Extracted, Invoice, LineItem, Party, Tax, Totals

LANGUAGES = ("en", "ru", "uk", "ka")

LABELS: dict[str, dict[str, str]] = {
    "en": {
        "title": "INVOICE",
        "number": "Invoice No",
        "date": "Date",
        "due": "Due date",
        "supplier": "Supplier",
        "buyer": "Bill to",
        "tax_id": "Tax ID",
        "item": "Item",
        "qty": "Qty",
        "unit": "Unit",
        "price": "Unit price",
        "total": "Total",
        "subtotal": "Subtotal",
        "vat": "VAT",
        "grand_total": "TOTAL DUE",
    },
    "ru": {
        "title": "СЧЁТ",
        "number": "Счёт №",
        "date": "Дата",
        "due": "Оплатить до",
        "supplier": "Поставщик",
        "buyer": "Покупатель",
        "tax_id": "ИНН",
        "item": "Наименование",
        "qty": "Кол-во",
        "unit": "Ед.",
        "price": "Цена",
        "total": "Сумма",
        "subtotal": "Итого",
        "vat": "НДС",
        "grand_total": "ВСЕГО К ОПЛАТЕ",
    },
    "uk": {
        "title": "РАХУНОК",
        "number": "Рахунок №",
        "date": "Дата",
        "due": "Сплатити до",
        "supplier": "Постачальник",
        "buyer": "Покупець",
        "tax_id": "ЄДРПОУ",
        "item": "Найменування",
        "qty": "К-сть",
        "unit": "Од.",
        "price": "Ціна",
        "total": "Сума",
        "subtotal": "Разом",
        "vat": "ПДВ",
        "grand_total": "ВСЬОГО ДО СПЛАТИ",
    },
    "ka": {
        "title": "ინვოისი",
        "number": "ინვოისი №",
        "date": "თარიღი",
        "due": "გადახდის ვადა",
        "supplier": "მიმწოდებელი",
        "buyer": "მყიდველი",
        "tax_id": "ს/კ",
        "item": "დასახელება",
        "qty": "რაოდ.",
        "unit": "ერთ.",
        "price": "ფასი",
        "total": "ჯამი",
        "subtotal": "შუალედური ჯამი",
        "vat": "დღგ",
        "grand_total": "სულ გადასახდელი",
    },
}

CURRENCIES = {"en": ("EUR", "USD", "GBP"), "ru": ("GEL", "UAH"), "uk": ("UAH",), "ka": ("GEL",)}
VAT_RATES = {
    "en": ("0.20", "0.19", "0.21"),
    "ru": ("0.20", "0.18"),
    "uk": ("0.20",),
    "ka": ("0.18",),
}
UNITS = {
    "en": ("pcs", "pack", "bottle"),
    "ru": ("шт", "уп", "фл"),
    "uk": ("шт", "уп", "фл"),
    "ka": ("ც", "შეკვრა", "ბოთლი"),
}

SUPPLIERS = {
    "en": (
        "Beauty Supplies Ltd",
        "Salon Pro Distribution",
        "Nordic Cosmetics AB",
        "Glow Wholesale",
    ),
    "ru": ("ООО Бьюти Трейд", "ИП Каримова А.Р.", "Профкосметика", "ООО Салон Сервис"),
    "uk": ("ТОВ Б'юті Трейд", "ФОП Коваленко О.В.", "Профкосметика Україна", "ТОВ Салон Сервіс"),
    "ka": ("შპს ბიუთი ტრეიდი", "ი/მ ნინო ბერიძე", "პროფკოსმეტიკა", "შპს სალონ სერვისი"),
}
BUYERS = {
    "en": ("Salon Nova", "Studio Luma", "The Hair Room", "Velvet Nails"),
    "ru": ("Салон Нова", "Студия Люма", "Барбершоп Точка", "Вельвет Нейлс"),
    "uk": ("Салон Нова", "Студія Люма", "Барбершоп Крапка", "Вельвет Нейлз"),
    "ka": ("სალონი ნოვა", "სტუდია ლუმა", "ბარბერშოპი წერტილი", "ველვეტ ნეილსი"),
}
ADDRESSES = {
    "en": ("12 Market Street, Manchester", "4 Rue des Fleurs, Lyon", "88 King Road, London"),
    "ru": ("ул. Руставели 12, Тбилиси", "пр. Свободы 4, Одесса", "ул. Мира 88, Батуми"),
    "uk": ("вул. Хрещатик 12, Київ", "просп. Свободи 4, Львів", "вул. Соборна 88, Дніпро"),
    "ka": ("რუსთაველის გამზ. 12, თბილისი", "აღმაშენებლის 4, ბათუმი", "ჭავჭავაძის 88, ქუთაისი"),
}
PRODUCTS = {
    "en": (
        "Argan shampoo 1L",
        "Keratin conditioner 1L",
        "Hair colour 7.1 60ml",
        "Nail gel base 15ml",
        "Disposable towels 50pc",
        "Barber clipper oil",
        "Bleach powder 500g",
        "Face mask sheet 10pc",
    ),
    "ru": (
        "Шампунь аргановый 1л",
        "Кондиционер кератин 1л",
        "Краска для волос 7.1 60мл",
        "База для геля 15мл",
        "Полотенца одноразовые 50шт",
        "Масло для машинки",
        "Осветляющая пудра 500г",
        "Маска тканевая 10шт",
    ),
    "uk": (
        "Шампунь аргановий 1л",
        "Кондиціонер кератин 1л",
        "Фарба для волосся 7.1 60мл",
        "База для гелю 15мл",
        "Рушники одноразові 50шт",
        "Олія для машинки",
        "Освітлююча пудра 500г",
        "Маска тканинна 10шт",
    ),
    "ka": (
        "არგანის შამპუნი 1ლ",
        "კერატინის კონდიციონერი 1ლ",
        "თმის საღებავი 7.1 60მლ",
        "გელის ბაზა 15მლ",
        "ერთჯერადი პირსახოცი 50ც",
        "საპარსი მანქანის ზეთი",
        "გამაღიავებელი ფხვნილი 500გ",
        "სახის ნიღაბი 10ც",
    ),
}

TWO_PLACES = Decimal("0.01")


@dataclass(frozen=True)
class InvoiceContent:
    language: str
    labels: dict[str, str]
    invoice: Invoice


def format_amount(value: Decimal, language: str) -> str:
    text = f"{value.quantize(TWO_PLACES):,.2f}"
    if language == "en":
        return text
    return text.replace(",", " ").replace(".", ",")


def format_date(value: date, language: str) -> str:
    return value.strftime("%d %b %Y") if language == "en" else value.strftime("%d.%m.%Y")


def _ex[T](value: T, quote: str) -> Extracted[T]:
    return Extracted[T](value=value, quote=quote, confidence=1.0)


def _tax_id(rng: random.Random, language: str) -> str:
    digits = {"en": 9, "ru": 10, "uk": 8, "ka": 9}[language]
    number = "".join(str(rng.randint(0, 9)) for _ in range(digits))
    return f"GB{number}" if language == "en" else number


def generate_content(rng: random.Random, language: str) -> InvoiceContent:
    if language not in LANGUAGES:
        raise ValueError(f"unknown language {language!r}")
    labels = LABELS[language]
    currency = rng.choice(CURRENCIES[language])
    issue = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
    due = issue + timedelta(days=rng.choice((7, 14, 30)))
    number = f"{rng.choice(('INV', 'F', 'A', 'SF'))}-{issue.year}-{rng.randint(1, 9999):04d}"

    items: list[LineItem] = []
    subtotal = Decimal(0)
    for name in rng.sample(PRODUCTS[language], k=rng.randint(2, 6)):
        quantity = Decimal(rng.randint(1, 12))
        unit_price = (Decimal(rng.randint(150, 9000)) / 100).quantize(TWO_PLACES)
        total = (quantity * unit_price).quantize(TWO_PLACES)
        subtotal += total
        items.append(
            LineItem(
                description=_ex(name, name),
                quantity=_ex(quantity, str(quantity)),
                unit=_ex(unit := rng.choice(UNITS[language]), unit),
                unit_price=_ex(unit_price, format_amount(unit_price, language)),
                total=_ex(total, format_amount(total, language)),
            )
        )

    rate = Decimal(rng.choice(VAT_RATES[language]))
    tax_amount = (subtotal * rate).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
    grand_total = subtotal + tax_amount
    rate_percent = f"{int(rate * 100)}%"

    def party(names: tuple[str, ...]) -> Party:
        name = rng.choice(names)
        tax_id = _tax_id(rng, language)
        address = rng.choice(ADDRESSES[language])
        return Party(
            name=_ex(name, name), tax_id=_ex(tax_id, tax_id), address=_ex(address, address)
        )

    invoice = Invoice(
        supplier=party(SUPPLIERS[language]),
        buyer=party(BUYERS[language]),
        number=_ex(number, number),
        issue_date=_ex(issue, format_date(issue, language)),
        due_date=_ex(due, format_date(due, language)),
        currency=_ex(currency, currency),
        language=_ex(language, language),
        line_items=items,
        taxes=[
            Tax(
                name=_ex(labels["vat"], labels["vat"]),
                rate=_ex(rate, rate_percent),
                amount=_ex(tax_amount, format_amount(tax_amount, language)),
            )
        ],
        totals=Totals(
            subtotal=_ex(subtotal, format_amount(subtotal, language)),
            tax_total=_ex(tax_amount, format_amount(tax_amount, language)),
            grand_total=_ex(grand_total, format_amount(grand_total, language)),
        ),
    )
    return InvoiceContent(language=language, labels=labels, invoice=invoice)
