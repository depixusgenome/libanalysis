import {NumberFormatter, StringFormatter} from "models/widgets/tables/cell_formatters"

export class DpxNumberFormatter extends NumberFormatter {
    doFormat(row, cell, value, columnDef, dataContext) {
        if(value == null || isNaN(value))
            return ""
        return super.doFormat(row, cell, value, columnDef, dataContext)
    }
    static initClass(): void {
        this.prototype.type = 'DpxNumberFormatter'
    }
}
DpxNumberFormatter.initClass()
