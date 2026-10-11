"""Czytelne (polskie) opisy cech finansowych używanych w aplikacji.

Słownik FEATURE_INFO obejmuje kolumny z V4FinBench, które najczęściej trafiają
do modelu. Dla kolumn spoza słownika `describe()` buduje opis automatycznie
z fragmentów nazwy (np. "Net_profit/total_assets" -> "Zysk netto / aktywa ogółem").
"""

import re

# Kolumny, których NIGDY nie wolno użyć jako cech: cel, flagi "z przyszłości",
# identyfikatory i metadane.
LEAKY_OR_ID_COLS = {
    'main_label', 'Insolvency_flag', 'insolvency_flag', 'operational_status',
    'num', 'emis_id', 'year', 'company', 'industry', 'link', 'country',
}

CATEGORIES = ['Płynność', 'Zadłużenie', 'Rentowność', 'Efektywność', 'Wzrost', 'Skala', 'Inne']

CATEGORY_ICONS = {
    'Płynność': '💧', 'Zadłużenie': '🏋️', 'Rentowność': '💰',
    'Efektywność': '⚙️', 'Wzrost': '📈', 'Skala': '🏢', 'Inne': '📋',
}

# higher_is_better: True -> wyższa wartość zwykle oznacza zdrowszą firmę
FEATURE_INFO = {
    'Current_assets/short_term_liabilities': dict(
        label='Płynność bieżąca', category='Płynność', higher_is_better=True,
        help='Ile razy aktywa obrotowe (zapasy, należności, gotówka) pokrywają długi do spłaty w ciągu roku. '
             'Poniżej 1 = firma może mieć problem z regulowaniem bieżących rachunków.'),
    'Current_assets-inventories/short_term_liabilities': dict(
        label='Płynność szybka', category='Płynność', higher_is_better=True,
        help='Jak płynność bieżąca, ale bez zapasów (które trudno szybko sprzedać). Bezpieczny poziom to ok. 1.'),
    'Current_assets-inventories-receivables/short_term_liabilities': dict(
        label='Płynność gotówkowa', category='Płynność', higher_is_better=True,
        help='Ile krótkoterminowych długów firma spłaciłaby od ręki samą gotówką.'),
    'Current_assets/current_liabilities': dict(
        label='Aktywa obrotowe / zobowiązania bieżące', category='Płynność', higher_is_better=True,
        help='Wariant wskaźnika płynności bieżącej.'),
    'Working_capital/total_assets': dict(
        label='Kapitał obrotowy / aktywa', category='Płynność', higher_is_better=True,
        help='Jaka część majątku to „wolny” kapitał obrotowy. Wartość ujemna to sygnał ostrzegawczy (składnik modelu Altmana).'),
    'Working_capital/equity': dict(
        label='Kapitał obrotowy / kapitał własny', category='Płynność', higher_is_better=True,
        help='Jaka część kapitału właścicieli finansuje bieżącą działalność.'),
    'Working_capital/total_operating_revenue': dict(
        label='Kapitał obrotowy / przychody', category='Płynność', higher_is_better=True,
        help='Zapas kapitału obrotowego w relacji do skali sprzedaży.'),
    'Working_capital/sales': dict(
        label='Kapitał obrotowy / sprzedaż', category='Płynność', higher_is_better=True,
        help='Zapas kapitału obrotowego w relacji do sprzedaży.'),
    'Inventories/working_capital': dict(
        label='Zapasy / kapitał obrotowy', category='Płynność', higher_is_better=False,
        help='Jak bardzo kapitał obrotowy jest „zamrożony” w zapasach.'),
    'Inventory/current_liabilities': dict(
        label='Zapasy / zobowiązania bieżące', category='Płynność', higher_is_better=False,
        help='Na ile spłata bieżących długów zależy od sprzedaży zapasów.'),
    'Equity/total_liabilities': dict(
        label='Kapitał własny / zobowiązania', category='Zadłużenie', higher_is_better=True,
        help='Ile złotych kapitału właścicieli przypada na złotówkę długu. Im więcej, tym bezpieczniej.'),
    'Equity_ratio': dict(
        label='Udział kapitału własnego', category='Zadłużenie', higher_is_better=True,
        help='Jaka część majątku jest sfinansowana przez właścicieli, a nie dług.'),
    'Equity/fixed_assets': dict(
        label='Pokrycie majątku trwałego kapitałem własnym', category='Zadłużenie', higher_is_better=True,
        help='Czy budynki, maszyny itp. są sfinansowane z kapitału właścicieli (zasada „złotej reguły bilansowej”).'),
    'Long_term_liabilities/equity': dict(
        label='Dług długoterminowy / kapitał własny', category='Zadłużenie', higher_is_better=False,
        help='Ile długu długoterminowego przypada na złotówkę kapitału właścicieli.'),
    'Long_term_liabilities/current_assets': dict(
        label='Dług długoterminowy / aktywa obrotowe', category='Zadłużenie', higher_is_better=False,
        help='Obciążenie długiem długoterminowym w relacji do majątku obrotowego.'),
    'Total_liabilities/EBITDA': dict(
        label='Dług / EBITDA', category='Zadłużenie', higher_is_better=False,
        help='Ile lat firma spłacałaby całe zadłużenie z wyniku operacyjnego. Powyżej ok. 4 to wysokie ryzyko.'),
    'Total_liabilities-cash/total_operating_revenue': dict(
        label='Dług netto / przychody', category='Zadłużenie', higher_is_better=False,
        help='Zadłużenie pomniejszone o gotówkę w relacji do rocznej sprzedaży.'),
    'Retained_profit/short_term_liabilities': dict(
        label='Zyski zatrzymane / długi krótkoterminowe', category='Zadłużenie', higher_is_better=True,
        help='Czy firma ma „poduszkę” z wypracowanych w przeszłości zysków.'),
    'Revenue/long_term_liabilities': dict(
        label='Przychody / dług długoterminowy', category='Zadłużenie', higher_is_better=True,
        help='Skala sprzedaży w relacji do długu długoterminowego.'),
    'Current_liabilities/sub_liabilities': dict(
        label='Udział długów krótkoterminowych', category='Zadłużenie', higher_is_better=False,
        help='Jaka część długów jest wymagalna w ciągu roku.'),
    'EBIT/total_assets': dict(
        label='Rentowność majątku (EBIT/aktywa)', category='Rentowność', higher_is_better=True,
        help='Ile zysku operacyjnego wypracowuje każda złotówka majątku (składnik modelu Altmana).'),
    'EBIT/total_operating_revenue': dict(
        label='Marża operacyjna', category='Rentowność', higher_is_better=True,
        help='Ile groszy zysku operacyjnego zostaje z każdej złotówki sprzedaży.'),
    'EBIT/total_liabilities': dict(
        label='Zysk operacyjny / zobowiązania', category='Rentowność', higher_is_better=True,
        help='Zdolność do obsługi długu z bieżącej działalności.'),
    'EBIT/equity': dict(
        label='Zysk operacyjny / kapitał własny', category='Rentowność', higher_is_better=True,
        help='Rentowność kapitału właścicieli na poziomie operacyjnym.'),
    'Net_profit/total_assets': dict(
        label='Rentowność aktywów (ROA)', category='Rentowność', higher_is_better=True,
        help='Ile zysku netto daje każda złotówka majątku firmy.'),
    'Net_profit/total_operating_revenue': dict(
        label='Marża netto', category='Rentowność', higher_is_better=True,
        help='Ile groszy czystego zysku zostaje z każdej złotówki przychodu.'),
    'Net_profit/current_assets': dict(
        label='Zysk netto / aktywa obrotowe', category='Rentowność', higher_is_better=True,
        help='Rentowność majątku obrotowego.'),
    'Net_profit/fixed_assets': dict(
        label='Zysk netto / majątek trwały', category='Rentowność', higher_is_better=True,
        help='Rentowność majątku trwałego.'),
    'Net_profit/sales_factor': dict(
        label='Marża netto (sprzedaż)', category='Rentowność', higher_is_better=True,
        help='Zysk netto w relacji do sprzedaży.'),
    'EBITDA_margin_factor': dict(
        label='Marża EBITDA', category='Rentowność', higher_is_better=True,
        help='Wynik operacyjny przed amortyzacją w relacji do przychodów.'),
    'Operating_expenses/sales': dict(
        label='Koszty operacyjne / sprzedaż', category='Efektywność', higher_is_better=False,
        help='Ile kosztuje wygenerowanie złotówki sprzedaży. Powyżej 1 = firma dokłada do działalności.'),
    'Total_operating_revenue/total_assets': dict(
        label='Rotacja aktywów', category='Efektywność', higher_is_better=True,
        help='Ile złotych sprzedaży generuje każda złotówka majątku (składnik modelu Altmana).'),
    'Revenue/fixed_assets_factor': dict(
        label='Przychody / majątek trwały', category='Efektywność', higher_is_better=True,
        help='Jak efektywnie wykorzystywane są maszyny, budynki itp.'),
    'Fixed_assets/total_assets': dict(
        label='Udział majątku trwałego', category='Efektywność', higher_is_better=None,
        help='Jaka część majątku to aktywa trwałe (zależy mocno od branży).'),
    'Receivables_turnover_days': dict(
        label='Cykl należności (dni)', category='Efektywność', higher_is_better=False,
        help='Średnio po ilu dniach klienci płacą firmie.'),
    'Cash_conversion_cycle_factor': dict(
        label='Cykl konwersji gotówki', category='Efektywność', higher_is_better=False,
        help='Jak długo pieniądze są zamrożone w zapasach i należnościach.'),
    'Equity/sales': dict(
        label='Kapitał własny / sprzedaż', category='Efektywność', higher_is_better=True,
        help='Kapitał właścicieli w relacji do skali sprzedaży.'),
    'Operating_revenue_growth': dict(
        label='Dynamika przychodów', category='Wzrost', higher_is_better=True,
        help='Zmiana przychodów rok do roku (1,00 = bez zmian, 1,20 = wzrost o 20%).'),
    'Receivables_growth': dict(
        label='Dynamika należności', category='Wzrost', higher_is_better=None,
        help='Zmiana należności rok do roku (1,00 = bez zmian).'),
    'Total_assets_growth': dict(
        label='Dynamika majątku', category='Wzrost', higher_is_better=True,
        help='Zmiana sumy bilansowej rok do roku (1,00 = bez zmian).'),
    'Inventories_growth': dict(
        label='Dynamika zapasów', category='Wzrost', higher_is_better=None,
        help='Zmiana zapasów rok do roku (1,00 = bez zmian).'),
    'Logarithm_of_total_operating_revenue': dict(
        label='Skala firmy (log przychodów)', category='Skala', higher_is_better=True,
        help='Logarytm rocznych przychodów — im wyższy, tym większa firma. Duże firmy rzadziej upadają.'),
    'Log_operating_profitability': dict(
        label='Skala zysku operacyjnego (log)', category='Skala', higher_is_better=True,
        help='Logarytm zysku operacyjnego.'),
    'Working_capital': dict(
        label='Kapitał obrotowy (kwota)', category='Skala', higher_is_better=True,
        help='Aktywa obrotowe minus zobowiązania krótkoterminowe, w walucie sprawozdania.'),
}

# Fragmenty nazw do automatycznego tłumaczenia kolumn spoza słownika
_TOKENS = [
    ('short_term_liabilities', 'zobowiązania krótkoterminowe'),
    ('long_term_liabilities', 'zobowiązania długoterminowe'),
    ('current_liabilities', 'zobowiązania bieżące'),
    ('total_liabilities', 'zobowiązania ogółem'),
    ('liabilities', 'zobowiązania'),
    ('total_operating_revenue', 'przychody operacyjne'),
    ('operating_revenue', 'przychody operacyjne'),
    ('total_assets', 'aktywa ogółem'),
    ('current_assets', 'aktywa obrotowe'),
    ('fixed_assets', 'majątek trwały'),
    ('working_capital', 'kapitał obrotowy'),
    ('retained_profit', 'zyski zatrzymane'),
    ('net_profit', 'zysk netto'),
    ('operating_expenses', 'koszty operacyjne'),
    ('inventories', 'zapasy'), ('inventory', 'zapasy'),
    ('receivables', 'należności'),
    ('equity', 'kapitał własny'),
    ('revenue', 'przychody'), ('sales', 'sprzedaż'),
    ('cash', 'gotówka'), ('growth', 'dynamika'),
    ('logarithm_of', 'log'), ('factor', ''), ('ratio', 'wskaźnik'),
]


def _auto_label(col):
    s = col
    for en, pl in _TOKENS:
        s = re.sub(en, pl, s, flags=re.IGNORECASE)
    s = s.replace('_', ' ').replace('/', ' / ').replace('-', ' − ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:1].upper() + s[1:]


def _auto_category(col):
    c = col.lower()
    if 'growth' in c:
        return 'Wzrost'
    if 'profit' in c or 'ebit' in c or 'margin' in c:
        return 'Rentowność'
    if 'log' in c:
        return 'Skala'
    if 'equity' in c or 'liabilities' in c:
        return 'Zadłużenie' if 'current_assets' not in c else 'Płynność'
    if 'current_assets' in c or 'working_capital' in c or 'cash' in c:
        return 'Płynność'
    if 'turnover' in c or 'expenses' in c or 'revenue' in c or 'sales' in c:
        return 'Efektywność'
    return 'Inne'


def describe(col):
    """Zwraca słownik: label, category, help, higher_is_better."""
    if col in FEATURE_INFO:
        info = dict(FEATURE_INFO[col])
    else:
        info = dict(label=_auto_label(col), category=_auto_category(col), higher_is_better=None,
                    help=f'Wskaźnik finansowy: {_auto_label(col)}.')
    info['column'] = col
    return info
