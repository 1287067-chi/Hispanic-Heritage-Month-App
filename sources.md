# Sources and Data Dictionary

**API:** World Bank Indicators API v2. No API key is required.

## World Bank data

The app uses the World Bank Indicators API to retrieve historical data for Mexico, Brazil, Argentina, Colombia, Chile, Peru, Venezuela, Ecuador, and Uruguay.

API format:

`https://api.worldbank.org/v2/country/{ISO3}/indicator/{INDICATOR_CODE}?date={START}:{END}&format=json&per_page=20000`

### Indicators

| Indicator | Code | Unit |
|---|---|---|
| Life expectancy at birth | `SP.DYN.LE00.IN` | Years |
| Unemployment, national estimate | `SL.UEM.TOTL.NE.ZS` | Percentage of labor force |
| GDP per capita | `NY.GDP.PCAP.CD` | Current US dollars per person |
| Crude birth rate | `SP.DYN.CBRT.IN` | Births per 1,000 people per year |
| Net migration | `SM.POP.NETM` | People, net |
| Intentional homicide rate | `VC.IHR.PSRC.P5` | Per 100,000 people |
| Consumer price inflation | `FP.CPI.TOTL.ZG` | Annual percentage change |
| Central government debt | `GC.DOD.TOTL.GD.ZS` | Percentage of GDP |
| Total reserves including gold | `FI.RES.TOTL.CD` | Current US dollars |
| External balance on goods and services | `NE.RSB.GNFS.ZS` | Percentage of GDP |

### World Bank documentation

- Indicators API documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/889386-about-the-indicators-api-documentation
- API basic call structures: https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures
- World Bank Open Data: https://data.worldbank.org/

## Peru gold production, 2000–2006

The World Bank indicators above do not directly measure annual gold production. In particular, `FI.RES.TOTL.CD` measures the monetary value of official reserves including gold, **not the amount of gold mined**.

For the project's gold-production research question, the app includes annual Peru gold-production values transcribed in rounded metric tonnes from Chart 7, “Gold production 1994 to 2010,” in *The Economic Contribution of Large-Scale Gold Mining in Peru*. The report is hosted by INGEMMET and credits INEI, Government of Peru, as the chart's source.

- Report: https://app.ingemmet.gob.pe/biblioteca/pdf/Econ-120.pdf
- USGS Minerals Yearbook: https://www.usgs.gov/centers/national-minerals-information-center/minerals-yearbook-metals-and-minerals
- World Mining Data: https://www.world-mining-data.info/
- Peru Ministry of Energy and Mines: https://www.gob.pe/minem

### Main finding

- **2000:** approximately 129 metric tonnes
- **2005:** approximately 210 metric tonnes
- **2006:** approximately 202 metric tonnes
- **Change from 2000 to 2006:** an increase of approximately 73 metric tonnes, or 56.6%.

Although production increased overall from 2000 to 2006, it declined from 2005 to 2006. The chart values are rounded and should not replace the original official series when exact precision is required. Mine production, refined output, and other gold measures are not necessarily interchangeable.

The chart's 2010 figure covers only the first nine months, so 2010 is excluded from the annual series in the app.

## Brazil comparison note

Brazil is included as a Latin American regional comparison. Portuguese is Brazil's official language; Brazil is included for regional comparison, not because it is Spanish-speaking.

## Data integrity

- The app does not fill missing World Bank observations with invented values.
- It reports available observations and data coverage.
- User-edited values represent scenarios and should not be described as official observations.
- World Bank data may be revised by the provider after retrieval.
