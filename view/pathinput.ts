import {build_views} from "core/build_views"
import * as p from "core/properties"
import {InputWidget, InputWidgetView} from "models/widgets/input_widget"

declare function jQuery(...args: any[]): any

export namespace PathInput {
    export type Attrs = p.AttrsOf<Props>
    export type Props = InputWidget.Props & {
        value:       p.Property<string>
        placeholder: p.Property<string>
        click:       p.Property<number>
    }
}

export interface PathInput extends PathInput.Attrs {}

export class PathInputView extends InputWidgetView {
    model: PathInput
    connect_signals(): void {
        super.connect_signals()
        this.connect(this.model.change, this.render)
    }

    render(): void {
        super.render()
        let label: string = ""
        if(this.model.title)
            label = "<label for="+this.model.id+"> "+this.model.title+" </label>"

        let txt = (
            "<input class='bk-widget-form-input' type='text'"
            +" id="+this.model.id
            +" name="+this.model.name
            +" value='"+this.model.value+"'"
            +" placeholder='"+this.model.placeholder
            +"' />")

        let btn = (
            "<button type='button'"
            +" class='bk-bs-btn bk-bs-btn-default'"
            +" style='margin-left:5px'><span class='icon-dpx-folder-plus'></span>"
            +"</button>"
        )

        jQuery(this.el).html("<div class='dpx-span'>"+label+txt+btn+"</div>")

        let elem = jQuery(this.el)

        let inp  = elem.find('input')
        if(this.model.height)
            // TODO - This 35 is a hack we should be able to compute it
            inp.height(this.model.height - 35)
        if(this.model.width)
            inp.width(this.model.width-50)

        inp.prop("disabled", this.model.disabled)
        inp.change(() => this.change_input())

        let btn = elem.find('button')
        btn.width(5)
        btn.prop("disabled", this.model.disabled)
        btn.click(() => this.change_click())
    }

    change_click(): void {
        this.model.click = this.model.click+1
    }

    change_input(): void {
        this.model.value = jQuery(this.el).find('input').val()
        super.change_input()
    }
}

export class PathInput extends InputWidget {
    properties: PathInput.Props
    constructor(attrs?: Partial<PathInput.Attrs>) { super(attrs); }
    static initClass(): void {
        this.prototype.type= "PathInput"
        this.prototype.default_view= PathInputView

        this.define({
            value:       [p.String, ""],
            placeholder: [p.String, ""],
            click:       [p.Number, 0],
        })
    }
}
PathInput.initClass()
