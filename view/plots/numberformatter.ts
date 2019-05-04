import {NumberFormatter} from "models/widgets/tables/cell_formatters"

export class DpxNumberFormatter extends NumberFormatter {
    doFormat(_row: any, _cell: any, value: any, _columnDef: any, _dataContext: any): string {
        if(value == null || isNaN(value))
            return ""
        return super.doFormat(_row, _cell, value, _columnDef, _dataContext)
    }
    static initClass(): void {
        this.prototype.type = 'DpxNumberFormatter'
    }
}
DpxNumberFormatter.initClass()
