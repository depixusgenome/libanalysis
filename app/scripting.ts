import *        as p    from "core/properties"
import {Model}          from "model"
import {DOMView} from "core/dom_view"

declare function jQuery(...args: any[]): any

export namespace DpxLoaded {
    export type Attrs = p.AttrsOf<Props>
    export type Props = Model.Props & {
        done: p.Property<number>
        resizedfig: p.Property<any>
    }
}

export interface DpxLoaded extends DpxLoaded.Attrs {}

export class DpxLoadedView extends DOMView {
    model: DpxLoaded
    connect_signals(): void {
        super().connect_signals()
        this.connect(
            this.model.properties.resizedfig.change,
            () => { this.model.on_resize_figure() }
        )
    }
}

export class DpxLoaded extends Model {
    properties: DpxLoaded.Props

    constructor(attrs?: Partial<DpxLoaded.Attrs>) {
        super(attrs);
        jQuery(() => { this.done = 1 })
    }

    static initClass(): void {
        this.prototype.default_view = DpxLoadedView
        this.prototype.type         = "DpxLoaded"
        this.define<DpxLoaded.Props>({
            done:       [p.Number, 0],
            resizedfig: [p.Instance, null],
        })
    }

    on_resize_figure(): void {
        if(this.resizedfig != null) {
            this.resizedfig.resize()
            this.resizedfig.layout()
        }
    }
}
