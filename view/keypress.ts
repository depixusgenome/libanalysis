import * as p  from "core/properties"
import {Model} from "model"
import {DOMView} from "core/dom_view"

declare function jQuery(...args: any[]): any

export class DpxKeyEventView extends DOMView {
    model: DpxKeyEvent
}

export namespace DpxKeyEvent {
    export type Attrs = p.AttrsOf<Props>

    export type Props = Model.Props & {
        keys:  p.Property<string[]>
        value: p.Property<string>
        count: p.Property<number>
    }
}


export interface DpxKeyEvent extends DpxKeyEvent.Attrs {}

export class DpxKeyEvent extends Model {
    properties: DpxKeyEvent.Props
    constructor(attrs?: Partial<DpxKeyEvent.Attrs>) {
        super(attrs);
        let fcn = (e: KeyboardEvent) => { this.dokeydown(e) }
        jQuery(document).keydown(fcn)
    }

    static initClass(): void {
        this.prototype.type = "DpxKeyEvent"
        this.prototype.default_view = DpxKeyEventView
        this.define<DpxKeyEvent.Props>({
            keys:  [p.Array, []],
            value: [p.String, ""],
            count: [p.Number, 0],
        })
    }

    dokeydown(evt: KeyboardEvent): void {
        let val: string = ""
        let tmp = {'alt': 'Alt', 'shift': 'Shift', 'ctrl': 'Control', 'meta': 'Meta'}
        for(let name in tmp)
            if(evt[name+'Key'])
                val += tmp[name]+"-"

        if (val == (evt.key+"-"))
            val = evt.key
        else
            val = val + evt.key

        if(this.keys.indexOf(val) > -1)
        {
            evt.preventDefault()
            evt.stopPropagation()
            this.value = val
            this.count = this.count+1
        }
    }
}
DpxKeyEvent.initClass()
