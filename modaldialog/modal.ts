import * as p               from "core/properties"
import {DOMView}            from "core/dom_view"
import {Model}              from "model"

declare function jQuery(...args: any[]): any
declare function __extends(...args:any[]): void
declare var _: any
declare var Bokeh: any

/* Modal was copied from backbone-modal */
function boundMethodCheck(instance:any, Constructor: any) {
    if (!(instance instanceof Constructor)) {
        throw new Error('Bound instance method accessed before binding');
    }
}
const indexOf = [].indexOf;

// @ts-ignore
let Modal: any = (function () {
    // Modal was copied from backbone-modal
    var Modal = /** @class */ (function (_super) {
        __extends(Modal, _super);
        function Modal(options:any) {
            if (options === void 0) { options = {}; }
            // @ts-ignore
            var _this = _super.call(this, options) || this;
            _this.rendererCompleted = _this.rendererCompleted.bind(_this);
            _this.checkKey = _this.checkKey.bind(_this);
            // check if the element on mouseup is not the modal itself
            _this.clickOutside = _this.clickOutside.bind(_this);
            _this.clickOutsideElement = _this.clickOutsideElement.bind(_this);
            _this.triggerView = _this.triggerView.bind(_this);
            _this.triggerSubmit = _this.triggerSubmit.bind(_this);
            _this.triggerCancel = _this.triggerCancel.bind(_this);
            _this.args = Array.prototype.slice.apply(arguments);
            // get all options
            _this.setUIElements();
            return _this;
        }
        Modal.prototype.render = function (options: any) {
            var data, ref;
            _super.prototype.render.call(this, options);
            // use openAt or overwrite this with your own functionality
            data = this.serializeData();
            if (!options || _.isEmpty(options)) {
                options = 0;
            }
            this.$el.addClass(this.prefix + "-wrapper");
            this.modalEl = jQuery('<div />').addClass(this.prefix + "-modal");
            if (this.template) {
                this.modalEl.html(this.buildTemplate(this.template, data));
            }
            this.$el.html(this.modalEl);
            if (this.viewContainer) {
                this.viewContainerEl = this.modalEl.find(this.viewContainer);
                this.viewContainerEl.addClass(this.prefix + "-modal__views");
            }
            else {
                this.viewContainerEl = this.modalEl;
            }
            // blur links to prevent double keystroke events
            jQuery(':focus').blur();
            if (((ref = this.views) != null ? ref.length : void 0) > 0 && this.showViewOnRender) {
                this.openAt(options);
            }
            if (typeof this.onRender === "function") {
                this.onRender();
            }
            this.delegateModalEvents();
            // show modal
            if (this.$el.fadeIn && this.animate) {
                this.modalEl.css({
                    opacity: 0
                });
                this.$el.fadeIn({
                    duration: 100,
                    complete: this.rendererCompleted
                });
            }
            else {
                this.rendererCompleted();
            }
            return this;
        };
        Modal.prototype.rendererCompleted = function () {
            var ref;
            boundMethodCheck(this, Modal);
            if (this.keyControl) {
                // global events for key and click outside the modal
                jQuery('body').on('keyup.bbm', this.checkKey);
                this.$el.on('mouseup.bbm', this.clickOutsideElement);
                this.$el.on('click.bbm', this.clickOutside);
            }
            this.modalEl.css({
                opacity: 1
            }).addClass(this.prefix + "-modal--open");
            if (typeof this.onShow === "function") {
                this.onShow();
            }
            return (ref = this.currentView) != null ? typeof ref.onShow === "function" ? ref.onShow() : void 0 : void 0;
        };
        Modal.prototype.setUIElements = function () {
            var ref;
            // get modal options
            this.template = this.getOption('template');
            this.views = this.getOption('views');
            if ((ref = this.views) != null) {
                ref.length = _.size(this.views);
            }
            this.viewContainer = this.getOption('viewContainer');
            this.animate = this.getOption('animate');
            // check if everything is right
            if (_.isUndefined(this.template) && _.isUndefined(this.views)) {
                throw new Error('No template or views defined for Modal');
            }
            if (this.template && this.views && _.isUndefined(this.viewContainer)) {
                throw new Error('No viewContainer defined for Modal');
            }
        };
        Modal.prototype.getOption = function (option: any) {
            // get class instance property
            if (!option) {
                return;
            }
            if (this.options && indexOf.call(this.options, option) >= 0 && (this.options[option] != null)) {
                return this.options[option];
            }
            else {
                return this[option];
            }
        };
        Modal.prototype.serializeData = function () {
            var data;
            // return the appropriate data for this view
            data = {};
            if (this.model) {
                data = _.extend(data, this.model.toJSON());
            }
            if (this.collection) {
                data = _.extend(data, {
                    items: this.collection.toJSON()
                });
            }
            return data;
        };
        Modal.prototype.delegateModalEvents = function () {
            var cancelEl, key, match, results, selector, submitEl, trigger;
            this.active = true;
            // get elements
            cancelEl = this.getOption('cancelEl');
            submitEl = this.getOption('submitEl');
            if (submitEl) {
                // set event handlers for submit and cancel
                this.$el.on('click', submitEl, this.triggerSubmit);
            }
            if (cancelEl) {
                this.$el.on('click', cancelEl, this.triggerCancel);
            }
            // set event handlers for views
            results = [];
            for (key in this.views) {
                if (_.isString(key) && key !== 'length') {
                    match = key.match(/^(\S+)\s*(.*)$/) as [any, any, any];
                    trigger = match[1];
                    selector = match[2];
                    results.push(this.$el.on(trigger, selector, this.views[key], this.triggerView));
                }
                else
                    results.push(void 0);
            }
            return results;
        };
        Modal.prototype.undelegateModalEvents = function () {
            var cancelEl, key, match, results, selector, submitEl, trigger;
            this.active = false;
            // get elements
            cancelEl = this.getOption('cancelEl');
            submitEl = this.getOption('submitEl');
            if (submitEl) {
                // remove event handlers for submit and cancel
                this.$el.off('click', submitEl, this.triggerSubmit);
            }
            if (cancelEl) {
                this.$el.off('click', cancelEl, this.triggerCancel);
            }
            // remove event handlers for views
            results = [];
            for (key in this.views) {
                if (_.isString(key) && key !== 'length') {
                    match = key.match(/^(\S+)\s*(.*)$/) as [any, any, any];
                    trigger = match[1];
                    selector = match[2];
                    results.push(this.$el.off(trigger, selector, this.views[key], this.triggerView));
                }
                else {
                    results.push(void 0);
                }
            }
            return results;
        };
        Modal.prototype.checkKey = function (e:any) {
            boundMethodCheck(this, Modal);
            if (this.active) {
                switch (e.keyCode) {
                    case 27:
                        return this.triggerCancel(e);
                    case 13:
                        return this.triggerSubmit(e);
                }
            }
        };
        Modal.prototype.clickOutside = function () {
            var ref;
            boundMethodCheck(this, Modal);
            if (((ref = this.outsideElement) != null ? ref.hasClass(this.prefix + "-wrapper") : void 0) && this.active) {
                return this.triggerCancel();
            }
        };
        Modal.prototype.clickOutsideElement = function (e: any) {
            boundMethodCheck(this, Modal);
            return this.outsideElement = jQuery(e.target);
        };
        Modal.prototype.buildTemplate = function (template: any, data: any) {
            var templateFunction;
            if (typeof template === 'function') {
                templateFunction = template;
            }
            else {
                templateFunction = _.template(jQuery(template).html());
            }
            return templateFunction(data);
        };
        Modal.prototype.buildView = function (viewType:any, options: any) {
            var view;
            // returns a DOMView instance, a function or an object
            if (!viewType) {
                return;
            }
            if (options && _.isFunction(options)) {
                options = options();
            }
            if (_.isFunction(viewType)) {
                view = new viewType(options || this.args[0]);
                if (view instanceof DOMView) {
                    return {
                        // @ts-ignore
                        el: view.render().$el,
                        view: view
                    };
                }
                else {
                    return {
                        el: viewType(options || this.args[0])
                    };
                }
            }
            return {
                view: viewType,
                el: viewType.$el
            };
        };
        Modal.prototype.triggerView = function (e:any) {
            var base, base1, index, instance, key, options, ref;
            boundMethodCheck(this, Modal);
            // trigger what view should be rendered
            if (e != null) {
                if (typeof e.preventDefault === "function") {
                    e.preventDefault();
                }
            }
            options = e.data;
            instance = this.buildView(options.view, options.viewOptions);
            if (this.currentView) {
                this.previousView = this.currentView;
                if (!((ref = options.openOptions) != null ? ref.skipSubmit : void 0)) {
                    if ((typeof (base = this.previousView).beforeSubmit === "function" ? base.beforeSubmit() : void 0) === false) {
                        return;
                    }
                    if (typeof (base1 = this.previousView).submit === "function") {
                        base1.submit();
                    }
                }
            }
            this.currentView = instance.view || instance.el;
            index = 0;
            for (key in this.views) {
                if (options.view === this.views[key].view) {
                    this.currentIndex = index;
                }
                index++;
            }
            if (options.onActive) {
                if (_.isFunction(options.onActive)) {
                    options.onActive(this);
                }
                else if (_.isString(options.onActive)) {
                    this[options.onActive].call(this, options);
                }
            }
            if (this.shouldAnimate) {
                return this.animateToView(instance.el);
            }
            else {
                this.shouldAnimate = true;
                return this.$(this.viewContainerEl).html(instance.el);
            }
        };
        Modal.prototype.animateToView = function (view:any) {
            var _this = this;
            var base, container, newHeight, previousHeight, ref, style, tester;
            style = {
                position: 'relative',
                top: -9999,
                left: -9999
            };
            tester = jQuery('<tester/>').css(style);
            tester.html(this.$el.clone().css(style));
            if (jQuery('tester').length !== 0) {
                jQuery('tester').replaceWith(tester);
            }
            else {
                jQuery('body').append(tester);
            }
            if (this.viewContainer) {
                container = tester.find(this.viewContainer);
            }
            else {
                container = tester.find("." + this.prefix + "-modal");
            }
            container.removeAttr('style');
            previousHeight = container.outerHeight();
            container.html(view);
            newHeight = container.outerHeight();
            if (previousHeight === newHeight) {
                this.$(this.viewContainerEl).html(view);
                if (typeof (base = this.currentView).onShow === "function") {
                    base.onShow();
                }
                return (ref = this.previousView) != null ? typeof ref.destroy === "function" ? ref.destroy() : void 0 : void 0;
            }
            else {
                if (this.animate) {
                    this.$(this.viewContainerEl).css({
                        opacity: 0
                    });
                    return this.$(this.viewContainerEl).animate({
                        height: newHeight
                    }, 100, function () {
                        var base1, ref1;
                        _this.$(_this.viewContainerEl).css({
                            opacity: 1
                        }).removeAttr('style');
                        _this.$(_this.viewContainerEl).html(view);
                        if (typeof (base1 = _this.currentView).onShow === "function") {
                            base1.onShow();
                        }
                        return (ref1 = _this.previousView) != null ? typeof ref1.destroy === "function" ? ref1.destroy() : void 0 : void 0;
                    });
                }
                else {
                    return this.$(this.viewContainerEl).css({
                        height: newHeight
                    }).html(view);
                }
            }
        };
        Modal.prototype.triggerSubmit = function (e:any) {
            var ref, ref1;
            boundMethodCheck(this, Modal);
            if (e != null) {
                e.preventDefault();
            }
            if (jQuery(e.target).is('textarea')) {
                return;
            }
            if (this.beforeSubmit) {
                if (this.beforeSubmit() === false) {
                    return;
                }
            }
            if (this.currentView && this.currentView.beforeSubmit) {
                if (this.currentView.beforeSubmit() === false) {
                    return;
                }
            }
            if (!this.submit && !((ref = this.currentView) != null ? ref.submit : void 0) && !this.getOption('submitEl')) {
                return this.triggerCancel();
            }
            if ((ref1 = this.currentView) != null) {
                if (typeof ref1.submit === "function") {
                    ref1.submit();
                }
            }
            if (typeof this.submit === "function") {
                this.submit();
            }
            if (this.regionEnabled) {
                return this.trigger('modal:destroy');
            }
            else {
                return this.destroy();
            }
        };
        Modal.prototype.triggerCancel = function (e:any) {
            boundMethodCheck(this, Modal);
            if (e != null) {
                e.preventDefault();
            }
            if (this.beforeCancel) {
                if (this.beforeCancel() === false) {
                    return;
                }
            }
            if (typeof this.cancel === "function") {
                this.cancel();
            }
            if (this.regionEnabled) {
                return this.trigger('modal:destroy');
            }
            else {
                return this.destroy();
            }
        };
        Modal.prototype.destroy = function () {
            var _this = this;
            let removeViews: any;
            jQuery('body').off('keyup.bbm', this.checkKey);
            this.$el.off('mouseup.bbm', this.clickOutsideElement);
            this.$el.off('click.bbm', this.clickOutside);
            jQuery('tester').remove();
            if (typeof this.onDestroy === "function") {
                this.onDestroy();
            }
            this.shouldAnimate = false;
            this.modalEl.addClass(this.prefix + "-modal--destroy");
            removeViews = function () {
                var ref;
                if ((ref = _this.currentView) != null) {
                    if (typeof ref.remove === "function") {
                        ref.remove();
                    }
                }
                return _this.remove();
            };
            if (this.$el.fadeOut && this.animate) {
                this.$el.fadeOut({
                    duration: 200
                });
                return _.delay(function () {
                    return removeViews();
                }, 200);
            }
            else {
                return removeViews();
            }
        };
        Modal.prototype.openAt = function (options:any) {
            var atIndex, attr, i, key, view;
            if (_.isNumber(options)) {
                atIndex = options;
            }
            else if (_.isNumber(options._index)) {
                atIndex = options._index;
            }
            i = 0;
            for (key in this.views) {
                if (key !== 'length') {
                    // go to specific index in views[]
                    if (_.isNumber(atIndex)) {
                        if (i === atIndex) {
                            view = this.views[key];
                        }
                        i++;
                        // use attributes to find a view in views[]
                    }
                    else if (_.isObject(options)) {
                        for (attr in this.views[key]) {
                            if (options[attr] === this.views[key][attr]) {
                                view = this.views[key];
                            }
                        }
                    }
                }
            }
            if (view) {
                this.currentIndex = _.indexOf(this.views, view);
                this.triggerView({
                    data: _.extend(view, {
                        openOptions: options
                    })
                });
            }
            return this;
        };
        Modal.prototype.next = function (options:any) {
            if (options === void 0) { options = {}; }
            if (this.currentIndex + 1 < this.views.length) {
                return this.openAt(_.extend(options, {
                    _index: this.currentIndex + 1
                }));
            }
        };
        Modal.prototype.previous = function (options:any) {
            if (options === void 0) { options = {}; }
            if (this.currentIndex - 1 < this.views.length - 1) {
                return this.openAt(_.extend(options, {
                    _index: this.currentIndex - 1
                }));
            }
        };
        return Modal;
    }(DOMView));
    ;
    Modal.prototype.prefix = 'bbm';
    Modal.prototype.animate = true;
    Modal.prototype.keyControl = true;
    Modal.prototype.showViewOnRender = true;
    return Modal;
}).call(this);

class DpxModalDialogView extends Modal {
    model: DpxModal
    startvalues: any
    constructor(attrs: any) { super(attrs); }
    template(data: any): string | null { return data['template'] }

    _form_values(): any {
        let vals: any = {}
        let ref:  any = jQuery(this.modalEl).find('#dpxbbmform').serializeArray();
        for(let i = 0; i < ref.length; ++i)
            vals[ref[i].name] = ref[i].value

        ref = jQuery(this.modalEl).find('#dpxbbmform input[type=checkbox]:not(:checked)')
        for(let i = 0; i < ref.length; ++i)
            vals[ref[i].name] = 'off'

        let elems = document.getElementsByClassName("bbm-dpx-curbtn")
        for(let i = 0; i < elems.length; ++i){
            let el = elems[i]
            if(el != null)
            {
                let attrv = el.getAttribute("tabvalue")
                let attrk = el.getAttribute("tabkey")
                if(attrv != null && attrk != null)
                    vals[attrk] = attrv
            }

        }
        return vals
    }

    render(): DpxModalDialogView {
        super.render()
        if(this.startvalues != null)
            this.startvalues = this._form_values()
        return this
    }

    cancel(): void { delete this.startvalues }

    beforeSubmit(): boolean
    { 
        // @ts-ignore
        return document.forms['dpxbbmform'].reportValidity() 
    }

    submit(): void {
        let vals = this._form_values()
        if(this.startvalues != null) {
            let tmp = vals;
            vals    = {}
            for(let key in tmp)
                if(this.startvalues[key] != tmp[key])
                    vals[key] = tmp[key]

            delete this.startvalues
        }

        this.model.results    = vals
        this.model.submitted += 1
        if(this.model.callback != null)
            this.model.callback.execute(this.model.results)
    }

    static initClass() : void 
    {
        this.prototype.cancelEl= '.dpx-modal-cancel'
        this.prototype.submitEl= '.dpx-modal-done'
    }
}
DpxModalDialogView.initClass()

export namespace DpxModal {
    export type Attrs = p.AttrsOf<Props>
    export type Props = Model.Props & {
        title:        p.Property<string>
        body:         p.Property<string>
        buttons:      p.Property<string>
        results:      p.Property<any>
        submitted:    p.Property<number>
        startdisplay: p.Property<number>
        keycontrol:   p.Property<boolean>
        callback:     p.Property<any>
    }
}

export interface DpxModal extends DpxModal.Attrs {}

export class DpxModalView extends DOMView {
    model: DpxModal
    initialize(): void {
        super.initialize()
        jQuery('body').append("<div class='dpx-modal-div'/>")
    }
    connect_signals(): void {
        super.connect_signals()
        this.connect(
            this.model.properties.startdisplay.change,
            () => this.model._startdisplaymodal()
        )
    }
}

export class DpxModal extends Model {
    properties: DpxModal.Props
    toJSON(){
        const bkclass  = Bokeh.version != '1.0.4' ? ' bk ' : ''
        let title:string = ""
        if (this.title == "")
            title = "<p style='height:5px;'></p>"
        else
            title = "<div class='bbm-modal__topbar'>"                           +
                        "<h3 class='bbm-modal__title'>"                         +
                            this.title                                          +
                        "</h3>"                                                 +
                    "</div>"

        let body  = "<div class='bbm-modal__section'>"                             +
            `<form id="dpxbbmform" class="${bkclass} bk-root">${this.body}</form>` +
                "</div>"

        let btns = ""
        if(this.buttons == "")
            btns  = `<div class='bbm-modal__bottombar ${bkclass} bk-root'>`         +
                `<button type='button' class='${bkclass} bk-btn bk-btn-default ` +
                        "dpx-modal-cancel'>Cancel</button>"                         +
                `<button type='button' class='${bkclass} bk-btn bk-btn-default ` +
                        "dpx-modal-done'>Apply</button>"                            +
                    "</div>"
        else
            btns  = `<div class='bbm-modal__bottombar ${bkclass} bk-root'>`         +
                        `<button type='button' class='${bkclass} bk-btn bk-btn-default `+
                        "dpx-modal-done'>"+this.buttons+"</button>"                     +
                    "</div>"
        return { template: "<fragment>"+title+" "+body+" "+btns+"</fragment>"}
    }

    clicktab(ind:number){
        let elems = document.getElementsByClassName("bbm-dpx-curtab")
        for(let i = 0; i < elems.length; ++i){
            let el = elems[i]
            if(el != null)
            {
                el.classList.add("bbm-dpx-hidden")
                el.classList.remove("bbm-dpx-curtab")
            }
        }

        elems = document.getElementsByClassName("bbm-dpx-curbtn")
        for(let i = 0; i < elems.length; ++i){
            let el = elems[i]
            if(el != null){
                el.classList.add("bbm-dpx-btn")
                el.classList.remove("bbm-dpx-curbtn")
                el.classList.remove("bk-active")
            }
        }

        let el = document.getElementById("bbm-dpx-tab-"+ind)
        if(el != null){
            el.classList.remove("bbm-dpx-hidden")
            el.classList.add("bbm-dpx-curtab")
        }

        el = document.getElementById("bbm-dpx-btn-"+ind)
        if(el != null){
            el.classList.remove("bbm-dpx-btn")
            el.classList.add("bbm-dpx-curbtn")
            el.classList.add("bk-active")
        }
    }

    _startdisplaymodal() {
        let mdl = new DpxModalDialogView({model: this})
        mdl.keyControl = this.keycontrol
        mdl.$el = jQuery(mdl.el)
        return jQuery('.dpx-modal-div').html(mdl.render().el)
    }

    static initClass(): void {
        this.prototype.default_view = DpxModalView
        this.prototype.type         = "DpxModal"
        this.define<DpxModal.Props>({
            title:        [p.String,  ""],
            body:         [p.String,  ""],
            buttons:      [p.String,  ""],
            results:      [p.Any,     {}],
            submitted:    [p.Number,  0],
            startdisplay: [p.Number,  0],
            keycontrol:   [p.Boolean, true],
            callback:     [p.Instance],
        })
    }
}
DpxModal.initClass()
