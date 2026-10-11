from liquiditylens.reconciliation import parse_inline_facts,reconstruct

def test_inline_source_parser_respects_scale_sign_currency_and_dimensions():
    html='''<html><xbrli:context id="entity"><xbrli:period><xbrli:startDate>2023-01-01</xbrli:startDate><xbrli:endDate>2023-03-31</xbrli:endDate></xbrli:period></xbrli:context>
    <xbrli:context id="segment"><xbrli:entity><xbrli:segment><xbrldi:explicitMember dimension="x">division</xbrldi:explicitMember></xbrli:segment></xbrli:entity><xbrli:period><xbrli:instant>2023-03-31</xbrli:instant></xbrli:period></xbrli:context>
    <xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit><xbrli:unit id="eur"><xbrli:measure>iso4217:EUR</xbrli:measure></xbrli:unit>
    <ix:nonFraction name="us-gaap:NetCashProvidedByUsedInOperatingActivities" contextRef="entity" unitRef="usd" scale="6" sign="-">1,250</ix:nonFraction>
    <ix:nonFraction name="us-gaap:Assets" contextRef="segment" unitRef="usd">999</ix:nonFraction>
    <ix:nonFraction name="us-gaap:Assets" contextRef="entity" unitRef="eur">999</ix:nonFraction></html>'''
    facts=parse_inline_facts(html)
    assert len(facts)==1 and facts[0]["value"]==-1_250_000_000
    assert facts[0]["start"]=="2023-01-01" and facts[0]["end"]=="2023-03-31"

def test_reconciliation_reconstructs_component_ytd_difference():
    p={"method":"ytd_difference_components","facts":[{"end":"2023-12-31","val":300},{"end":"2023-09-30","val":200},{"end":"2023-09-30","val":20}]}
    assert reconstruct(p)==80

def test_external_instance_format_is_explicitly_unsupported():
    assert parse_inline_facts('<html><p>Traditional filing with a separate XBRL instance.</p></html>')==[]

