### Context
Discrete catalog stats were published as fixed-bin histograms; Knowball needed one league shape so inclusive CDF and chart y-axis match.

### Decision
Stage D fits a reflected Gaussian KDE for every catalog `stat.id` (same grid / catalog bound reflection as the former continuous path) and writes `curve`. Catalog `kind` is copied as metadata only. League JSON omits `bins` and `samples`.

### Consequences
- Stage E percentiles always use `_kde_cdf` (knowball `kdeCdf`). Discrete percentiles change; republish pages with league files.
- Do not branch Stage D on `kind`. Catalog `bin_width` is unused for densities.
- Never re-embed league curves on player pages.
